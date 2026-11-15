"""Turn stat history into a per-player distribution of possible scores.

Each player becomes three numbers the simulator can draw from:

    p_active   probability the player posts a real score at all
    ln_mu      log-normal location of the "he played" component
    ln_sigma   log-normal scale of that component

A log-normal is the right shape for fantasy scoring: non-negative, right-skewed,
the occasional huge week. The zero spike (inactive, benched, a goose egg) is a
separate Bernoulli, because mixing it into one bell curve understates both the
duds and the ceilings -- and the ceiling is exactly what an underdog needs.
"""

from __future__ import annotations

import json
import math
import sqlite3
from statistics import NormalDist
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..db import PLAYER_STAT_COLUMNS, TEAM_STAT_COLUMNS
from .scoring import (
    DEFAULT_SLOT_COUNTS,
    PPR_RULES,
    apply_rules,
    dst_stats_frame,
    offense_stats_frame,
)

# Positions we model. Everything else on a fantasy roster is a D/ST.
OFFENSE_POSITIONS = ("QB", "RB", "FB", "WR", "TE", "K")

# Model hyper-parameters. Deliberately few, and all in one place so the
# backtest can be used to argue about them.
HALF_LIFE_GAMES = 4.0        # recent form decays by half every 4 games
LOOKBACK_GAMES = 10          # games of history per player
SHRINK_GAMES = 3.0           # pseudo-games of positional prior mixed in
DEF_SHRINK_GAMES = 5.0       # pseudo-games mixed into a defence's effect
DEF_FACTOR_CLAMP = (0.75, 1.30)
DUD_THRESHOLD = 1.0          # a game at or under this counts as a zero
MIN_CV = 0.25                # floor on the active component's spread
SEASONS_BACK = 1             # prior seasons mixed in alongside the current one

# Mean multiplier by ESPN injury designation: the chance the player suits up.
AVAILABILITY: dict[str, float] = {
    "ACTIVE": 1.0,
    "NORMAL": 1.0,
    "QUESTIONABLE": 0.88,
    "DAY_TO_DAY": 0.88,
    "DOUBTFUL": 0.35,
    "OUT": 0.0,
    "INJURY_RESERVE": 0.0,
    "SUSPENSION": 0.0,
    "NOT_ACTIVE": 0.0,
    "BYE": 0.0,
}


@dataclass
class PlayerProjection:
    """One player's simulated-score distribution, plus the why behind it."""

    key: str
    name: str
    position: str
    team: str | None
    opponent: str | None
    p_active: float
    ln_mu: float
    ln_sigma: float
    mean: float
    sd: float
    n_games: int
    status: str
    def_factor: float = 1.0
    note: str = ""

    @property
    def params(self) -> tuple[float, float, float]:
        return (self.p_active, self.ln_mu, self.ln_sigma)

    @property
    def floor_is_zero(self) -> bool:
        return self.p_active <= 0.0

    def quantile(self, q: float) -> float:
        """Quantile of the full mixture, zero spike included.

        Reporting a mean and a standard deviation for a right-skewed
        distribution with an atom at zero is misleading; a 10th-90th range is
        what a manager actually wants to see.
        """
        if self.p_active <= 0.0:
            return 0.0
        zero_mass = 1.0 - self.p_active
        if q <= zero_mass:
            return 0.0
        u = (q - zero_mass) / self.p_active
        u = min(max(u, 1e-6), 1 - 1e-6)
        return float(math.exp(self.ln_mu + self.ln_sigma * NormalDist().inv_cdf(u)))


ZERO = PlayerProjection(
    key="", name="", position="", team=None, opponent=None,
    p_active=0.0, ln_mu=0.0, ln_sigma=0.0, mean=0.0, sd=0.0, n_games=0, status="OUT",
)


# --- league settings ------------------------------------------------------
def load_league_settings(
    conn: sqlite3.Connection, league_id: str, season: int
) -> tuple[dict[str, float], dict[int, int], str]:
    row = conn.execute(
        "SELECT name, scoring_json, slots_json FROM leagues WHERE league_id = ? AND season = ?",
        (league_id, season),
    ).fetchone()
    if row is None:
        return dict(PPR_RULES), dict(DEFAULT_SLOT_COUNTS), f"League {league_id}"
    rules = {**PPR_RULES, **json.loads(row["scoring_json"] or "{}")}
    slots = {int(k): int(v) for k, v in json.loads(row["slots_json"] or "{}").items()}
    return rules, (slots or dict(DEFAULT_SLOT_COUNTS)), (row["name"] or f"League {league_id}")


def week_opponents(conn: sqlite3.Connection, season: int, week: int) -> dict[str, str]:
    """team -> opponent for one week. Teams absent from the map are on bye."""
    rows = conn.execute(
        "SELECT home_team, away_team FROM games WHERE season = ? AND week = ? AND game_type = 'REG'",
        (season, week),
    ).fetchall()
    out: dict[str, str] = {}
    for row in rows:
        out[row["home_team"]] = row["away_team"]
        out[row["away_team"]] = row["home_team"]
    return out


# --- scored history -------------------------------------------------------
def scored_history(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    rules: Mapping[str, float],
    *,
    seasons_back: int = SEASONS_BACK,
) -> pd.DataFrame:
    """Every game strictly before (season, week), scored under ``rules``.

    Strictly-before is the whole point: it is what makes a backtest honest.
    """
    first_season = season - seasons_back
    cols = ", ".join(PLAYER_STAT_COLUMNS)
    players = pd.read_sql_query(
        f"SELECT player_id AS key, position, season, week, team, opponent, {cols}"
        " FROM player_weeks"
        " WHERE season_type = 'REG' AND position IN ({})"
        "   AND season >= ? AND (season < ? OR week < ?)".format(
            ",".join("?" * len(OFFENSE_POSITIONS))
        ),
        conn,
        params=(*OFFENSE_POSITIONS, first_season, season, week),
    )
    if not players.empty:
        players["points"] = apply_rules(offense_stats_frame(players), rules)
        players["position"] = players["position"].replace({"FB": "RB"})
        players = players[["key", "position", "season", "week", "team", "opponent", "points"]]

    tcols = ", ".join(TEAM_STAT_COLUMNS)
    teams = pd.read_sql_query(
        f"SELECT team, season, week, opponent, points_allowed, {tcols}"
        " FROM team_weeks WHERE season_type = 'REG' AND season >= ? AND (season < ? OR week < ?)",
        conn,
        params=(first_season, season, week),
    )
    if not teams.empty:
        teams["points"] = apply_rules(dst_stats_frame(teams), rules)
        teams["key"] = "DST_" + teams["team"].astype(str)
        teams["position"] = "D/ST"
        teams = teams[["key", "position", "season", "week", "team", "opponent", "points"]]

    frames = [f for f in (players, teams) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["key", "position", "season", "week", "team", "opponent", "points"])
    hist = pd.concat(frames, ignore_index=True)
    # A single global ordering makes "last N games" a simple tail().
    hist["order"] = hist["season"] * 100 + hist["week"]
    return hist.sort_values("order").reset_index(drop=True)


# --- priors ---------------------------------------------------------------
@dataclass
class Priors:
    """Positional baselines and defence adjustments, fitted on history."""

    position_mean: dict[str, float] = field(default_factory=dict)
    position_cv: dict[str, float] = field(default_factory=dict)
    position_dud: dict[str, float] = field(default_factory=dict)
    defense: dict[tuple[str, str], float] = field(default_factory=dict)

    def mean_for(self, position: str) -> float:
        return self.position_mean.get(position, 6.0)

    def cv_for(self, position: str) -> float:
        return max(self.position_cv.get(position, 0.6), MIN_CV)

    def dud_for(self, position: str) -> float:
        return min(max(self.position_dud.get(position, 0.1), 0.0), 0.9)

    def defense_factor(self, opponent: str | None, position: str) -> float:
        if not opponent:
            return 1.0
        return self.defense.get((opponent, position), 1.0)


def fit_priors(hist: pd.DataFrame) -> Priors:
    """Positional baselines, and how much each defence suppresses each position."""
    priors = Priors()
    if hist.empty:
        return priors

    # Only count games where the player was actually fantasy-relevant, so the
    # baseline is not dragged to zero by deep-bench appearances.
    played = hist[hist["points"] > DUD_THRESHOLD]
    for position, grp in played.groupby("position", observed=True):
        mean = float(grp["points"].mean())
        sd = float(grp["points"].std(ddof=0) or 0.0)
        priors.position_mean[position] = mean
        priors.position_cv[position] = (sd / mean) if mean > 0 else 0.6
    for position, grp in hist.groupby("position", observed=True):
        priors.position_dud[position] = float((grp["points"] <= DUD_THRESHOLD).mean())

    # How much a defence gives up to a position, relative to the league. Summed
    # per game so a defence facing three good WRs is not rewarded for it.
    off = hist[(hist["position"] != "D/ST") & hist["opponent"].notna()]
    if not off.empty:
        per_game = (
            off.groupby(["opponent", "position", "season", "week"], observed=True)["points"]
            .sum()
            .reset_index()
        )
        league = per_game.groupby("position", observed=True)["points"].mean()
        agg = per_game.groupby(["opponent", "position"], observed=True)["points"].agg(["mean", "count"])
        lo, hi = DEF_FACTOR_CLAMP
        for (team, position), row in agg.iterrows():
            base = float(league.get(position, 0.0))
            if base <= 0:
                continue
            raw = float(row["mean"]) / base
            n = float(row["count"])
            # Shrink toward 1.0: one quiet game against a defence proves little.
            shrunk = (n * raw + DEF_SHRINK_GAMES) / (n + DEF_SHRINK_GAMES)
            priors.defense[(team, position)] = float(min(max(shrunk, lo), hi))
    return priors


# --- per-player fitting ---------------------------------------------------
def _lognormal_params(mean: float, var: float) -> tuple[float, float]:
    """Moment-match a log-normal to a positive mean and variance."""
    mean = max(mean, 1e-6)
    var = max(var, (MIN_CV * mean) ** 2)
    sigma_sq = math.log1p(var / (mean * mean))
    sigma = math.sqrt(max(sigma_sq, 1e-6))
    mu = math.log(mean) - 0.5 * sigma_sq
    return mu, sigma


def _weights(n: int) -> np.ndarray:
    """Exponential recency weights, newest last."""
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)
    return 0.5 ** (ages / HALF_LIFE_GAMES)


def fit_player(
    key: str,
    name: str,
    position: str,
    games: pd.DataFrame,
    priors: Priors,
    *,
    team: str | None = None,
    opponent: str | None = None,
    status: str = "ACTIVE",
) -> PlayerProjection:
    """Fit one player's distribution from their recent games."""
    availability = AVAILABILITY.get((status or "ACTIVE").upper(), 1.0)
    if opponent is None:
        availability = 0.0
        status = "BYE"
    if availability <= 0.0:
        return PlayerProjection(
            key=key, name=name, position=position, team=team, opponent=opponent,
            p_active=0.0, ln_mu=0.0, ln_sigma=0.0, mean=0.0, sd=0.0,
            n_games=int(len(games)), status=status,
            note="bye week" if status == "BYE" else f"out ({status.lower()})",
        )

    recent = games.tail(LOOKBACK_GAMES)
    points = recent["points"].to_numpy(dtype=np.float64) if len(recent) else np.zeros(0)
    n = len(points)
    prior_mean = priors.mean_for(position)
    prior_cv = priors.cv_for(position)
    prior_dud = priors.dud_for(position)

    if n == 0:
        dud = prior_dud
        active_mean = prior_mean
        active_var = (prior_cv * prior_mean) ** 2
    else:
        w = _weights(n)
        w_sum = float(w.sum())
        dud_mask = points <= DUD_THRESHOLD
        raw_dud = float(w[dud_mask].sum() / w_sum)
        # Shrink the dud rate toward the positional base rate too.
        dud = (n * raw_dud + SHRINK_GAMES * prior_dud) / (n + SHRINK_GAMES)
        live = points[~dud_mask]
        live_w = w[~dud_mask]
        if live.size and live_w.sum() > 0:
            m = float((live_w * live).sum() / live_w.sum())
            v = float((live_w * (live - m) ** 2).sum() / live_w.sum())
        else:
            m, v = prior_mean, (prior_cv * prior_mean) ** 2
        n_eff = float(live.size)
        active_mean = (n_eff * m + SHRINK_GAMES * prior_mean) / (n_eff + SHRINK_GAMES)
        blended_var = (n_eff * v + SHRINK_GAMES * (prior_cv * prior_mean) ** 2) / (n_eff + SHRINK_GAMES)
        active_var = max(blended_var, (MIN_CV * active_mean) ** 2)

    # The matchup adjustment scales the active component and keeps its shape.
    factor = priors.defense_factor(opponent, position)
    active_mean *= factor
    active_var *= factor * factor

    p_active = max(0.0, min(1.0, (1.0 - min(max(dud, 0.0), 0.95)) * availability))
    ln_mu, ln_sigma = _lognormal_params(active_mean, active_var)

    # Report the distribution's actual moments, mixture and all.
    second = p_active * (active_var + active_mean * active_mean)
    mean = p_active * active_mean
    sd = math.sqrt(max(second - mean * mean, 0.0))
    return PlayerProjection(
        key=key, name=name, position=position, team=team, opponent=opponent,
        p_active=p_active, ln_mu=ln_mu, ln_sigma=ln_sigma, mean=mean, sd=sd,
        n_games=n, status=status, def_factor=factor,
        note="" if n else "no history: positional prior",
    )


def project_roster(
    hist: pd.DataFrame,
    priors: Priors,
    roster: Sequence[Mapping[str, object]],
    opponents: Mapping[str, str],
) -> list[PlayerProjection]:
    """Project every player on a roster for one week.

    ``roster`` rows need ``player_id``/``key``, ``name``, ``position``,
    ``pro_team`` and ``injury_status``.
    """
    by_key = {k: g for k, g in hist.groupby("key", observed=True)} if not hist.empty else {}
    empty = pd.DataFrame(columns=["points"])
    out: list[PlayerProjection] = []
    for row in roster:
        key = str(row.get("player_id") or row.get("key") or "")
        position = str(row.get("position") or "")
        if position == "FB":
            position = "RB"
        team = row.get("pro_team") or row.get("team")
        team = str(team) if team else None
        status = str(row.get("injury_status") or "ACTIVE")
        opponent = opponents.get(team) if team else None
        out.append(
            fit_player(
                key=key,
                name=str(row.get("name") or key),
                position=position,
                games=by_key.get(key, empty),
                priors=priors,
                team=team,
                opponent=opponent,
                status=status,
            )
        )
    return out


def params_matrix(projections: Sequence[PlayerProjection]) -> np.ndarray:
    """Pack projections into the (P, 3) float32 array the backends consume."""
    out = np.zeros((len(projections), 3), dtype=np.float32)
    for i, p in enumerate(projections):
        out[i] = (p.p_active, p.ln_mu, p.ln_sigma)
    return out
