"""Replay past weeks and check whether the probabilities hold up.

For each historical matchup the model is rebuilt using only games played
*before* that week, both sides are given the lineup a normal manager would set
(their projected best), and the simulator states a win probability. Then the
real stat lines for that week decide who actually won.

That gives a stack of (stated probability, what happened) pairs, which is
exactly what a calibration curve needs. The comparison point is a projected-
points forecast: projections give a margin, and a fitted logistic turns that
margin into a probability. If the simulator is not better than that, it is not
earning its keep.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..db import PLAYER_STAT_COLUMNS, TEAM_STAT_COLUMNS
from ..model.scoring import apply_rules, dst_stats_frame, offense_stats_frame
from ..sim.backends import get_backend
from ..sim.engine import (
    WeekModel,
    analytic_moments,
    build_week_model,
    pick_opponent_lineup,
)
from ..sim.lineups import eligibility_from_rows
from ..model.projections import params_matrix


def actual_points(
    conn: sqlite3.Connection, season: int, week: int, rules: dict[str, float]
) -> dict[str, float]:
    """What every player and defence really scored in one week, under ``rules``."""
    cols = ", ".join(PLAYER_STAT_COLUMNS)
    players = pd.read_sql_query(
        f"SELECT player_id AS key, {cols} FROM player_weeks"
        " WHERE season = ? AND week = ? AND season_type = 'REG'",
        conn,
        params=(season, week),
    )
    out: dict[str, float] = {}
    if not players.empty:
        points = apply_rules(offense_stats_frame(players), rules)
        out.update(dict(zip(players["key"].astype(str), points.astype(float))))

    tcols = ", ".join(TEAM_STAT_COLUMNS)
    teams = pd.read_sql_query(
        f"SELECT team, points_allowed, {tcols} FROM team_weeks"
        " WHERE season = ? AND week = ? AND season_type = 'REG'",
        conn,
        params=(season, week),
    )
    if not teams.empty:
        points = apply_rules(dst_stats_frame(teams), rules)
        out.update({f"DST_{t}": float(v) for t, v in zip(teams["team"].astype(str), points)})
    return out


@dataclass
class MatchupRecord:
    season: int
    week: int
    home_team_id: int
    away_team_id: int
    p_home_model: float
    projected_margin: float
    home_projected: float
    away_projected: float
    home_actual: float
    away_actual: float
    home_won: float            # 1, 0, or 0.5 for an exact tie
    n_missing_scores: int


def _best_lineup(spec, slots, projections, rows, max_lineups: int) -> np.ndarray:
    """The lineup a projected-points manager would set: highest expected total."""
    return pick_opponent_lineup(spec, slots, projections, rows, "best", max_lineups=max_lineups)


def _score_lineup(
    projections: Sequence, lineup: np.ndarray, truth: dict[str, float]
) -> tuple[float, int]:
    total, missing = 0.0, 0
    for i in lineup.tolist():
        key = projections[i].key
        if key in truth:
            total += truth[key]
        else:
            # A player with no stat line that week did not play: that is a real
            # zero, not a gap in the data. Counted so it can be audited.
            missing += 1
    return total, missing


def backtest_week(
    conn: sqlite3.Connection,
    league_id: str,
    season: int,
    week: int,
    *,
    backend: str = "numpy",
    n_sims: int = 200_000,
    seed: int = 0,
    max_lineups: int = 200_000,
    model: WeekModel | None = None,
) -> list[MatchupRecord]:
    """Replay every matchup in one week."""
    model = model or build_week_model(conn, league_id, season, week)
    truth = actual_points(conn, season, week, model.rules)
    simulate = get_backend(backend)

    matchups = conn.execute(
        "SELECT home_team_id, away_team_id FROM matchups WHERE league_id = ? AND season = ? AND week = ?",
        (league_id, season, week),
    ).fetchall()

    records: list[MatchupRecord] = []
    for row in matchups:
        home_id, away_id = int(row["home_team_id"]), int(row["away_team_id"])
        home_rows = model.roster(conn, home_id)
        away_rows = model.roster(conn, away_id)
        if not home_rows or not away_rows:
            continue
        home_proj = model.project(home_rows)
        away_proj = model.project(away_rows)
        home_spec = eligibility_from_rows(home_rows)
        away_spec = eligibility_from_rows(away_rows)

        home_lineup = _best_lineup(home_spec, model.slots, home_proj, home_rows, max_lineups)
        away_lineup = _best_lineup(away_spec, model.slots, away_proj, away_rows, max_lineups)

        outcome = simulate(
            params_matrix(home_proj),
            params_matrix(away_proj),
            home_lineup.reshape(1, -1),
            away_lineup,
            n_sims,
            seed + week,
        )
        p_home = float(outcome.win_prob[0])

        home_mean, _ = analytic_moments(home_proj, home_lineup.reshape(1, -1))
        away_mean, _ = analytic_moments(away_proj, away_lineup.reshape(1, -1))
        home_score, miss_h = _score_lineup(home_proj, home_lineup, truth)
        away_score, miss_a = _score_lineup(away_proj, away_lineup, truth)

        if home_score > away_score:
            won = 1.0
        elif home_score < away_score:
            won = 0.0
        else:
            won = 0.5

        records.append(
            MatchupRecord(
                season=season,
                week=week,
                home_team_id=home_id,
                away_team_id=away_id,
                p_home_model=p_home,
                projected_margin=float(home_mean[0] - away_mean[0]),
                home_projected=float(home_mean[0]),
                away_projected=float(away_mean[0]),
                home_actual=home_score,
                away_actual=away_score,
                home_won=won,
                n_missing_scores=miss_h + miss_a,
            )
        )
    return records


def run_backtest(
    conn: sqlite3.Connection,
    league_id: str,
    seasons: Iterable[int],
    weeks: Iterable[int],
    *,
    backend: str = "numpy",
    n_sims: int = 200_000,
    seed: int = 0,
    max_lineups: int = 200_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Replay a range of seasons and weeks. Returns one row per matchup."""
    records: list[MatchupRecord] = []
    for season in seasons:
        for week in weeks:
            has_matchups = conn.execute(
                "SELECT 1 FROM matchups WHERE league_id = ? AND season = ? AND week = ? LIMIT 1",
                (league_id, season, week),
            ).fetchone()
            if not has_matchups:
                continue
            model = build_week_model(conn, league_id, season, week)
            week_records = backtest_week(
                conn, league_id, season, week,
                backend=backend, n_sims=n_sims, seed=seed, max_lineups=max_lineups, model=model,
            )
            records.extend(week_records)
            if progress:
                print(f"  {season} week {week:>2}: {len(week_records)} matchups", flush=True)
    return pd.DataFrame([r.__dict__ for r in records])


# --- does the lineup choice actually win more games? ---------------------
@dataclass
class DecisionRecord:
    """One manager's lineup decision for one week, scored both ways."""

    season: int
    week: int
    team_id: int
    differs: bool
    p_win_chosen: float
    p_win_points: float
    points_chosen: float      # projected points of the win-probability lineup
    points_points: float      # projected points of the points-optimal lineup
    actual_chosen: float
    actual_points: float
    opponent_actual: float
    won_chosen: float
    won_points: float


def decision_backtest(
    conn: sqlite3.Connection,
    league_id: str,
    seasons: Iterable[int],
    weeks: Iterable[int],
    *,
    backend: str = "numpy",
    n_sims: int = 100_000,
    seed: int = 0,
    max_lineups: int = 200_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Compare the two decision rules on the weeks that actually happened.

    For every team in every week: rank all legal lineups by win probability,
    and separately by projected points. Where the two rules pick different
    lineups, score both against what the opponent really put up. This is the
    only question that matters in practice -- not whether the probabilities are
    calibrated, but whether acting on them wins more often.
    """
    from ..sim.engine import optimise_week

    records: list[DecisionRecord] = []
    for season in seasons:
        for week in weeks:
            rows = conn.execute(
                "SELECT home_team_id, away_team_id FROM matchups"
                " WHERE league_id = ? AND season = ? AND week = ?",
                (league_id, season, week),
            ).fetchall()
            if not rows:
                continue
            model = build_week_model(conn, league_id, season, week)
            truth = actual_points(conn, season, week, model.rules)

            for row in rows:
                for team_id, opp_id in (
                    (int(row["home_team_id"]), int(row["away_team_id"])),
                    (int(row["away_team_id"]), int(row["home_team_id"])),
                ):
                    try:
                        advice = optimise_week(
                            conn, league_id, season, week, team_id,
                            opponent_id=opp_id, backend=backend, n_sims=n_sims,
                            seed=seed + week, max_lineups=max_lineups, week_model=model,
                        )
                    except (LookupError, ValueError):
                        continue

                    i, j = advice.best_by_win_prob, advice.best_by_points
                    chosen = advice.lineups[i]
                    points_pick = advice.lineups[j]
                    actual_chosen, _ = _score_lineup(advice.projections, chosen, truth)
                    actual_points_pick, _ = _score_lineup(advice.projections, points_pick, truth)
                    # The opponent really does set their projected-best lineup.
                    opp_actual, _ = _score_lineup(
                        advice.opp_projections, advice.opp_lineup, truth
                    )

                    def outcome(mine: float) -> float:
                        return 1.0 if mine > opp_actual else (0.5 if mine == opp_actual else 0.0)

                    records.append(
                        DecisionRecord(
                            season=season,
                            week=week,
                            team_id=team_id,
                            differs=advice.they_disagree,
                            p_win_chosen=float(advice.win_prob[i]),
                            p_win_points=float(advice.win_prob[j]),
                            points_chosen=float(advice.lineup_mean[i]),
                            points_points=float(advice.lineup_mean[j]),
                            actual_chosen=actual_chosen,
                            actual_points=actual_points_pick,
                            opponent_actual=opp_actual,
                            won_chosen=outcome(actual_chosen),
                            won_points=outcome(actual_points_pick),
                        )
                    )
            if progress:
                print(f"  {season} week {week:>2}: {len(rows) * 2} decisions", flush=True)
    return pd.DataFrame([r.__dict__ for r in records])
