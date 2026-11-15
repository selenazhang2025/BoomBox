"""Put the pieces together: roster in, ranked lineups out.

The output people actually want is not a projection. It is: of every lineup I am
allowed to set, which one wins this matchup most often, and how much does each
start/sit decision move that number.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from ..model.projections import (
    PlayerProjection,
    fit_priors,
    load_league_settings,
    params_matrix,
    project_roster,
    scored_history,
    week_opponents,
)
from ..model.scoring import SLOT_LABELS, starting_slots
from .backends import SimOutcome, get_backend
from .lineups import RosterSpec, assign_slots, eligibility_from_rows, enumerate_with_trim

OpponentMode = Literal["best", "actual"]
DEFAULT_SIMS = 1_000_000


# --- results --------------------------------------------------------------
@dataclass
class Advice:
    """Everything the CLI and the charts need from one week's optimisation."""

    league_name: str
    season: int
    week: int
    team_name: str
    opponent_name: str
    slots: list[int]
    spec: RosterSpec
    projections: list[PlayerProjection]
    opp_projections: list[PlayerProjection]
    lineups: np.ndarray                 # (L, K) roster indices
    win_prob: np.ndarray                # (L,)
    lineup_mean: np.ndarray             # (L,) projected points
    lineup_sd: np.ndarray               # (L,)
    opp_lineup: np.ndarray
    opp_mean: float
    opp_sd: float
    outcome: SimOutcome
    dropped: list[int] = field(default_factory=list)

    # --- rankings ---------------------------------------------------------
    @property
    def best_by_win_prob(self) -> int:
        return int(np.argmax(self.win_prob))

    @property
    def best_by_points(self) -> int:
        return int(np.argmax(self.lineup_mean))

    @property
    def they_disagree(self) -> bool:
        a = sorted(self.lineups[self.best_by_win_prob].tolist())
        b = sorted(self.lineups[self.best_by_points].tolist())
        return a != b

    def order_by_win_prob(self) -> np.ndarray:
        return np.argsort(-self.win_prob, kind="stable")

    def slot_assignment(self, lineup_index: int) -> list[tuple[int, int]]:
        return assign_slots(self.spec, self.lineups[lineup_index].tolist(), self.slots)

    def describe_lineup(self, lineup_index: int) -> list[tuple[str, str, float]]:
        """(slot label, player name, projected mean) for one lineup."""
        out = []
        for slot, player in self.slot_assignment(lineup_index):
            out.append((SLOT_LABELS.get(slot, str(slot)), self.spec.names[player], self.projections[player].mean))
        return out

    # --- start/sit swings -------------------------------------------------
    def player_swings(self) -> list[tuple[int, float, float, bool]]:
        """How much each player is worth, in win probability.

        For every player: the best win probability achievable with them in the
        lineup, minus the best achievable without them. Positive means benching
        them costs you that much; negative means starting them does.

        Returns (player index, win-prob swing, points swing, starts in the
        optimal lineup), sorted by the size of the swing.
        """
        n_players = len(self.spec)
        best_with = np.full(n_players, -np.inf)
        best_without = np.full(n_players, -np.inf)
        best_with_pts = np.full(n_players, -np.inf)
        best_without_pts = np.full(n_players, -np.inf)

        membership = np.zeros((len(self.lineups), n_players), dtype=bool)
        rows = np.repeat(np.arange(len(self.lineups)), self.lineups.shape[1])
        membership[rows, self.lineups.ravel()] = True

        for i in range(n_players):
            inc = membership[:, i]
            if inc.any():
                best_with[i] = self.win_prob[inc].max()
                best_with_pts[i] = self.lineup_mean[inc].max()
            if (~inc).any():
                best_without[i] = self.win_prob[~inc].max()
                best_without_pts[i] = self.lineup_mean[~inc].max()

        optimal = set(self.lineups[self.best_by_win_prob].tolist())
        out: list[tuple[int, float, float, bool]] = []
        for i in range(n_players):
            if not np.isfinite(best_with[i]) or not np.isfinite(best_without[i]):
                continue  # a player who must start, or who can never start
            out.append(
                (
                    i,
                    float(best_with[i] - best_without[i]),
                    float(best_with_pts[i] - best_without_pts[i]),
                    i in optimal,
                )
            )
        out.sort(key=lambda row: -abs(row[1]))
        return out


# --- loading --------------------------------------------------------------
def load_roster_rows(
    conn: sqlite3.Connection, league_id: str, season: int, week: int, team_id: int
) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM roster_slots WHERE league_id = ? AND season = ? AND week = ? AND team_id = ?",
        (league_id, season, week, team_id),
    ).fetchall()
    return [dict(r) for r in rows]


def team_names(conn: sqlite3.Connection, league_id: str, season: int) -> dict[int, str]:
    rows = conn.execute(
        "SELECT team_id, name FROM fantasy_teams WHERE league_id = ? AND season = ?", (league_id, season)
    ).fetchall()
    return {int(r["team_id"]): r["name"] for r in rows}


def find_opponent(
    conn: sqlite3.Connection, league_id: str, season: int, week: int, team_id: int
) -> int:
    row = conn.execute(
        "SELECT home_team_id, away_team_id FROM matchups"
        " WHERE league_id = ? AND season = ? AND week = ? AND (home_team_id = ? OR away_team_id = ?)",
        (league_id, season, week, team_id, team_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"no week {week} matchup for team {team_id}")
    return int(row["away_team_id"] if int(row["home_team_id"]) == team_id else row["home_team_id"])


def analytic_moments(
    projections: Sequence[PlayerProjection], lineups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of each lineup's total, assuming players are independent.

    This is the "projected points" column, and it is free -- no simulation
    needed. The whole point of the project is that ranking by this disagrees
    with ranking by win probability.
    """
    means = np.array([p.mean for p in projections], dtype=np.float64)
    variances = np.array([p.sd ** 2 for p in projections], dtype=np.float64)
    lineup_mean = means[lineups].sum(axis=1)
    lineup_sd = np.sqrt(variances[lineups].sum(axis=1))
    return lineup_mean, lineup_sd


def pick_opponent_lineup(
    spec: RosterSpec,
    slots: Sequence[int],
    projections: Sequence[PlayerProjection],
    rows: Sequence[dict],
    mode: OpponentMode = "best",
    *,
    max_lineups: int = 200_000,
) -> np.ndarray:
    """Which lineup to assume the opponent sets."""
    means = [p.mean for p in projections]
    if mode == "actual":
        started = [
            i
            for i, row in enumerate(rows)
            if row.get("lineup_slot_id") is not None and int(row["lineup_slot_id"]) not in (20, 21)
        ]
        if len(started) == len(slots):
            return np.array(sorted(started), dtype=np.int32)
        # Fall through to "best" when the league has not set its lineup yet.
    candidates, _ = enumerate_with_trim(spec, list(slots), means, max_lineups=max_lineups)
    lineup_mean, _ = analytic_moments(projections, candidates)
    return candidates[int(np.argmax(lineup_mean))].astype(np.int32)


# --- one week's model, built once ----------------------------------------
@dataclass
class WeekModel:
    """Everything about a week that does not depend on which team you are.

    Fitting the priors means loading and scoring every prior game, which is far
    too expensive to redo for each of a league's five matchups -- let alone for
    every matchup of a multi-season backtest. So it is built once per week.
    """

    league_id: str
    league_name: str
    season: int
    week: int
    rules: dict[str, float]
    slots: list[int]
    history: "object"           # pandas DataFrame of scored prior games
    priors: object              # Priors
    opponents: dict[str, str]
    team_names: dict[int, str]

    def roster(self, conn: sqlite3.Connection, team_id: int) -> list[dict]:
        return load_roster_rows(conn, self.league_id, self.season, self.week, team_id)

    def project(self, rows: Sequence[dict]) -> list[PlayerProjection]:
        return project_roster(self.history, self.priors, rows, self.opponents)


def build_week_model(
    conn: sqlite3.Connection, league_id: str, season: int, week: int
) -> WeekModel:
    rules, slot_counts, league_name = load_league_settings(conn, league_id, season)
    slots = starting_slots(slot_counts)
    if not slots:
        raise ValueError("league has no starting slots configured")
    history = scored_history(conn, season, week, rules)
    return WeekModel(
        league_id=league_id,
        league_name=league_name,
        season=season,
        week=week,
        rules=rules,
        slots=slots,
        history=history,
        priors=fit_priors(history),
        opponents=week_opponents(conn, season, week),
        team_names=team_names(conn, league_id, season),
    )


# --- the main entry point -------------------------------------------------
def optimise_week(
    conn: sqlite3.Connection,
    league_id: str,
    season: int,
    week: int,
    team_id: int,
    *,
    opponent_id: int | None = None,
    backend: str = "numpy",
    n_sims: int = DEFAULT_SIMS,
    seed: int = 0,
    opponent_mode: OpponentMode = "best",
    max_lineups: int = 200_000,
    backend_kwargs: dict | None = None,
    week_model: "WeekModel | None" = None,
) -> Advice:
    """Rank every legal lineup for one team in one week by win probability."""
    model = week_model or build_week_model(conn, league_id, season, week)
    slots = model.slots

    opponent_id = opponent_id if opponent_id is not None else find_opponent(conn, league_id, season, week, team_id)
    my_rows = model.roster(conn, team_id)
    opp_rows = model.roster(conn, opponent_id)
    if not my_rows or not opp_rows:
        raise LookupError(f"no roster stored for week {week} (teams {team_id} and {opponent_id})")

    mine = model.project(my_rows)
    theirs = model.project(opp_rows)

    spec = eligibility_from_rows(my_rows)
    opp_spec = eligibility_from_rows(opp_rows)

    lineups, dropped = enumerate_with_trim(
        spec, slots, [p.mean for p in mine], max_lineups=max_lineups
    )
    opp_lineup = pick_opponent_lineup(
        opp_spec, slots, theirs, opp_rows, opponent_mode, max_lineups=max_lineups
    )

    simulate = get_backend(backend)
    outcome = simulate(
        params_matrix(mine),
        params_matrix(theirs),
        lineups,
        opp_lineup,
        n_sims,
        seed,
        **(backend_kwargs or {}),
    )

    lineup_mean, lineup_sd = analytic_moments(mine, lineups)
    opp_mean, opp_sd = analytic_moments(theirs, opp_lineup.reshape(1, -1))
    names = model.team_names

    return Advice(
        league_name=model.league_name,
        season=season,
        week=week,
        team_name=names.get(team_id, f"Team {team_id}"),
        opponent_name=names.get(opponent_id, f"Team {opponent_id}"),
        slots=list(slots),
        spec=spec,
        projections=mine,
        opp_projections=theirs,
        lineups=lineups,
        win_prob=outcome.win_prob,
        lineup_mean=lineup_mean,
        lineup_sd=lineup_sd,
        opp_lineup=opp_lineup,
        opp_mean=float(opp_mean[0]),
        opp_sd=float(opp_sd[0]),
        outcome=outcome,
        dropped=dropped,
    )
