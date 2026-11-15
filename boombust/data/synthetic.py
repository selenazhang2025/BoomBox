"""Build a plausible fake ESPN league out of real nflverse players.

This exists so the whole pipeline -- rosters, slots, matchups, optimisation,
backtesting -- runs and is testable without anyone's ESPN credentials. The
players, their stat lines and the outcomes are all real; only the league is
invented. Fully deterministic given a seed.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..db import PLAYER_STAT_COLUMNS, session
from ..model.scoring import (
    DEFAULT_SLOT_COUNTS,
    PPR_RULES,
    apply_rules,
    offense_stats_frame,
)

# Positions a fantasy roster is drafted from, and how many of each.
ROSTER_PLAN: tuple[tuple[str, int], ...] = (
    ("QB", 2),
    ("RB", 5),
    ("WR", 6),
    ("TE", 2),
    ("K", 1),
    ("D/ST", 1),
)

# Which ESPN lineup slots each position can fill, bench (20) included.
ELIGIBLE_SLOTS: dict[str, list[int]] = {
    "QB": [0, 7, 20],
    "RB": [2, 3, 23, 7, 20],
    "WR": [3, 4, 5, 23, 7, 20],
    "TE": [5, 6, 23, 7, 20],
    "K": [17, 20],
    "D/ST": [16, 20],
}

TEAM_NAMES: tuple[str, ...] = (
    "Boom or Bust", "Regression Candidates", "Standard Deviants", "Expected Value",
    "Variance Enjoyers", "Tail Risk", "Monte Carlo Drivers", "Point Estimates",
    "Heavy Tails", "Prior Beliefs", "Posterior Odds", "Sample Space",
)


def _draft_pool(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Rank draftable players by the *previous* season's production.

    Using the prior season keeps the draft free of information from the season
    we are about to simulate, which matters for the backtest.
    """
    cols = ", ".join(PLAYER_STAT_COLUMNS)
    prior = season - 1
    df = pd.read_sql_query(
        f"SELECT pw.player_id, p.name, pw.position, {cols}"
        " FROM player_weeks pw JOIN players p ON p.player_id = pw.player_id"
        " WHERE pw.season = ? AND pw.season_type = 'REG'"
        "   AND pw.position IN ('QB','RB','FB','WR','TE','K')",
        conn,
        params=(prior,),
    )
    if df.empty:  # no prior season in the database: fall back to this one
        df = pd.read_sql_query(
            f"SELECT pw.player_id, p.name, pw.position, {cols}"
            " FROM player_weeks pw JOIN players p ON p.player_id = pw.player_id"
            " WHERE pw.season = ? AND pw.season_type = 'REG'"
            "   AND pw.position IN ('QB','RB','FB','WR','TE','K')",
            conn,
            params=(season,),
        )
    if df.empty:
        raise RuntimeError(f"no player data available to draft season {season}")
    df["points"] = apply_rules(offense_stats_frame(df), PPR_RULES)
    df["position"] = df["position"].replace({"FB": "RB"})
    agg = (
        df.groupby(["player_id", "name", "position"], as_index=False, observed=True)["points"]
        .sum()
        .sort_values("points", ascending=False)
    )
    # Keep only players who are on a roster in the season we are building.
    active = pd.read_sql_query(
        "SELECT DISTINCT player_id, team FROM player_weeks WHERE season = ? AND season_type = 'REG'",
        conn,
        params=(season,),
    )
    agg = agg.merge(active.drop_duplicates("player_id"), on="player_id", how="inner")
    return agg.reset_index(drop=True)


def _dst_pool(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    teams = pd.read_sql_query(
        "SELECT DISTINCT team FROM team_weeks WHERE season = ? AND season_type = 'REG' ORDER BY team",
        conn,
        params=(season,),
    )
    teams["player_id"] = "DST_" + teams["team"]
    teams["name"] = teams["team"] + " D/ST"
    teams["position"] = "D/ST"
    return teams[["player_id", "name", "position", "team"]]


def _snake_order(n_teams: int, n_rounds: int) -> list[int]:
    order: list[int] = []
    for rnd in range(n_rounds):
        picks = list(range(n_teams))
        if rnd % 2:
            picks.reverse()
        order.extend(picks)
    return order


def _round_robin(n_teams: int, weeks: Sequence[int]) -> list[tuple[int, int, int]]:
    """Circle-method schedule: (week, home_team_id, away_team_id), 1-indexed ids."""
    ids = list(range(1, n_teams + 1))
    if n_teams % 2:
        ids.append(-1)  # bye marker
    half = len(ids) // 2
    rotation = ids[:]
    out: list[tuple[int, int, int]] = []
    for i, week in enumerate(weeks):
        pairs = list(zip(rotation[:half], reversed(rotation[half:])))
        for j, (a, b) in enumerate(pairs):
            if a == -1 or b == -1:
                continue
            home, away = (a, b) if (i + j) % 2 == 0 else (b, a)
            out.append((week, home, away))
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]
    return out


def build_league(
    season: int,
    *,
    league_id: str = "synthetic",
    n_teams: int = 10,
    weeks: Sequence[int] | None = None,
    seed: int = 7,
    db_path: Path | str | None = None,
    slot_counts: dict[int, int] | None = None,
) -> dict[str, int]:
    """Draft a league from real players and write it to the database."""
    weeks = list(weeks) if weeks is not None else list(range(1, 15))
    slot_counts = dict(slot_counts or DEFAULT_SLOT_COUNTS)
    rng = np.random.default_rng(seed)

    with session(db_path) as conn:
        pool = _draft_pool(conn, season)
        dst = _dst_pool(conn, season)

        # Draft position by position, snaking, with a little noise on the board
        # so the "best" team is not simply the top of every ranking.
        rosters: dict[int, list[dict]] = {t: [] for t in range(1, n_teams + 1)}
        for position, per_team in ROSTER_PLAN:
            board = dst if position == "D/ST" else pool[pool["position"] == position]
            board = board.copy()
            need = per_team * n_teams
            if len(board) < need:
                # Silently handing some teams a short roster would produce
                # illegal lineups later, with a confusing error a long way from
                # the cause. Fail here instead.
                raise RuntimeError(
                    f"not enough {position}s to draft {n_teams} teams for {season}: "
                    f"need {need}, pool has {len(board)}"
                )
            board = board.head(max(need * 2, need))
            jitter = rng.normal(0.0, 0.35, size=len(board))
            board = board.assign(rank_noise=np.arange(len(board)) + jitter).sort_values("rank_noise")
            picks = board.head(need).to_dict("records")
            for pick, team_id in zip(picks, _snake_order(n_teams, per_team)):
                rosters[team_id + 1].append({**pick, "position": position})

        conn.execute(
            "INSERT OR REPLACE INTO leagues (league_id, season, name, current_week, scoring_json, slots_json, source)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                league_id, season, f"BoomBust Synthetic {season}", max(weeks),
                json.dumps(PPR_RULES), json.dumps({str(k): v for k, v in slot_counts.items()}),
                "synthetic",
            ),
        )
        for team_id in range(1, n_teams + 1):
            conn.execute(
                "INSERT OR REPLACE INTO fantasy_teams (league_id, season, team_id, name, owner) VALUES (?,?,?,?,?)",
                (league_id, season, team_id, TEAM_NAMES[(team_id - 1) % len(TEAM_NAMES)], f"owner{team_id}"),
            )

        # Rosters are static across weeks: no synthetic waiver wire. Real
        # availability still varies, because byes come from the real schedule.
        rows = 0
        for week in weeks:
            for team_id, players in rosters.items():
                for p in players:
                    conn.execute(
                        "INSERT OR REPLACE INTO roster_slots (league_id, season, week, team_id, espn_player_id,"
                        " player_id, name, position, pro_team, eligible_slots, lineup_slot_id, injury_status)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            league_id, season, week, team_id, str(p["player_id"]), str(p["player_id"]),
                            p["name"], p["position"], p.get("team"),
                            json.dumps(ELIGIBLE_SLOTS[p["position"]]), None, "ACTIVE",
                        ),
                    )
                    rows += 1

        schedule = _round_robin(n_teams, weeks)
        for week, home, away in schedule:
            conn.execute(
                "INSERT OR REPLACE INTO matchups (league_id, season, week, home_team_id, away_team_id,"
                " home_points, away_points, home_projected, away_projected) VALUES (?,?,?,?,?,?,?,?,?)",
                (league_id, season, week, home, away, None, None, None, None),
            )

    return {"teams": n_teams, "roster_slots": rows, "matchups": len(schedule)}
