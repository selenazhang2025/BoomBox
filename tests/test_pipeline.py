"""End-to-end: a database of stat lines in, a ranked set of lineups out.

Builds a small but complete database from fabricated stat lines -- no network,
no ESPN -- then runs the same path the CLI runs: draft a league, fit the model,
enumerate lineups, simulate, and replay the week as a backtest.
"""

from __future__ import annotations

import numpy as np
import pytest

from boombust.data.synthetic import build_league
from boombust.db import PLAYER_STAT_COLUMNS, TEAM_STAT_COLUMNS, init_db, session
from boombust.sim.engine import build_week_model, find_opponent, optimise_week
from boombust.validate.backtest import actual_points, backtest_week, decision_backtest

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
# Deep enough that a four-team league can be drafted in full, with headroom.
ROSTER = [("QB", 10), ("RB", 24), ("WR", 28), ("TE", 10), ("K", 6)]
SEASONS = (2022, 2023)
WEEKS = range(1, 7)


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    """A complete miniature league universe, deterministic from one seed."""
    path = tmp_path_factory.mktemp("boombust") / "test.sqlite"
    init_db(path)
    rng = np.random.default_rng(99)

    players = []
    for position, count in ROSTER:
        for i in range(count):
            players.append(
                {
                    "player_id": f"{position}{i:02d}",
                    "name": f"{position} Player {i}",
                    "position": position,
                    "skill": float(rng.uniform(0.5, 1.6)),
                }
            )

    with session(path) as conn:
        for p in players:
            conn.execute(
                "INSERT OR REPLACE INTO players (player_id, name, position, last_team, last_season)"
                " VALUES (?,?,?,?,?)",
                (p["player_id"], p["name"], p["position"], TEAMS[0], max(SEASONS)),
            )

        stat_for = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards",
                    "TE": "receiving_yards", "K": "fg_made_30_39"}
        for season in SEASONS:
            for week in WEEKS:
                # Pair the eight NFL teams up for the week.
                order = list(TEAMS)
                for a, b in zip(order[::2], order[1::2]):
                    game_id = f"{season}_{week:02d}_{a}_{b}"
                    conn.execute(
                        "INSERT OR REPLACE INTO games (game_id, season, week, game_type, home_team,"
                        " away_team, home_score, away_score) VALUES (?,?,?,'REG',?,?,?,?)",
                        (game_id, season, week, a, b,
                         float(rng.integers(10, 35)), float(rng.integers(10, 35))),
                    )
                for i, team in enumerate(TEAMS):
                    opponent = TEAMS[i + 1] if i % 2 == 0 else TEAMS[i - 1]
                    conn.execute(
                        "INSERT OR REPLACE INTO team_weeks (team, season, week, season_type, opponent,"
                        f" points_allowed, {', '.join(TEAM_STAT_COLUMNS)})"
                        f" VALUES (?,?,?,'REG',?,?,{','.join('?' * len(TEAM_STAT_COLUMNS))})",
                        (team, season, week, opponent, float(rng.integers(3, 38)),
                         *[float(rng.integers(0, 4)) for _ in TEAM_STAT_COLUMNS]),
                    )
                for j, p in enumerate(players):
                    team = TEAMS[j % len(TEAMS)]
                    opponent = TEAMS[(j % len(TEAMS)) + 1] if (j % len(TEAMS)) % 2 == 0 else TEAMS[(j % len(TEAMS)) - 1]
                    values = {c: 0.0 for c in PLAYER_STAT_COLUMNS}
                    scale = 55.0 if p["position"] == "QB" else 22.0
                    values[stat_for[p["position"]]] = float(
                        max(rng.normal(p["skill"] * scale, scale * 0.45), 0.0)
                    )
                    if p["position"] in ("WR", "TE", "RB"):
                        values["receptions"] = float(rng.integers(0, 9))
                    conn.execute(
                        f"INSERT OR REPLACE INTO player_weeks (player_id, season, week, season_type,"
                        f" team, opponent, position, {', '.join(PLAYER_STAT_COLUMNS)})"
                        f" VALUES (?,?,?,'REG',?,?,?,{','.join('?' * len(PLAYER_STAT_COLUMNS))})",
                        (p["player_id"], season, week, team, opponent, p["position"],
                         *[values[c] for c in PLAYER_STAT_COLUMNS]),
                    )

    build_league(2023, league_id="test", n_teams=4, weeks=list(WEEKS), seed=3, db_path=path)
    return path


def test_a_pool_too_shallow_to_draft_fails_loudly(db):
    """A short roster would surface much later as an unfillable slot."""
    with pytest.raises(RuntimeError, match="not enough"):
        build_league(2023, league_id="too-big", n_teams=12, weeks=[1], seed=1, db_path=db)


def test_the_league_was_drafted_with_full_legal_rosters(db):
    with session(db) as conn:
        rows = conn.execute(
            "SELECT team_id, COUNT(*) n FROM roster_slots"
            " WHERE league_id='test' AND season=2023 AND week=3 GROUP BY team_id"
        ).fetchall()
    assert len(rows) == 4
    assert all(r["n"] == 17 for r in rows), "every team should get the full roster plan"


def test_no_player_was_drafted_by_two_teams(db):
    with session(db) as conn:
        rows = conn.execute(
            "SELECT player_id, COUNT(DISTINCT team_id) n FROM roster_slots"
            " WHERE league_id='test' AND season=2023 AND week=3 GROUP BY player_id"
        ).fetchall()
    assert all(r["n"] == 1 for r in rows)


def test_optimise_week_produces_a_coherent_ranking(db):
    with session(db) as conn:
        advice = optimise_week(conn, "test", 2023, 4, 1, n_sims=40_000, seed=1)

    assert len(advice.lineups) > 1
    assert advice.lineups.shape[1] == len(advice.slots)
    assert 0.0 <= advice.win_prob.min() and advice.win_prob.max() <= 1.0

    best = advice.best_by_win_prob
    assert advice.win_prob[best] == advice.win_prob.max()
    # Every starter is a real, distinct player on this roster.
    starters = advice.lineups[best].tolist()
    assert len(set(starters)) == len(starters)
    assert max(starters) < len(advice.spec)
    # The slot assignment must fill every slot.
    assert len(advice.slot_assignment(best)) == len(advice.slots)
    # Projected points and win probability should broadly agree, even where
    # they pick different lineups.
    assert np.corrcoef(advice.win_prob, advice.lineup_mean)[0, 1] > 0.8


def test_the_two_sides_of_a_matchup_give_complementary_probabilities(db):
    """P(A beats B) and P(B beats A) have to add to one."""
    with session(db) as conn:
        opponent = find_opponent(conn, "test", 2023, 4, 1)
        model = build_week_model(conn, "test", 2023, 4)
        a = optimise_week(conn, "test", 2023, 4, 1, opponent_id=opponent,
                          n_sims=300_000, seed=5, week_model=model)
        b = optimise_week(conn, "test", 2023, 4, opponent, opponent_id=1,
                          n_sims=300_000, seed=5, week_model=model)
    # Each side is simulated against the other's projected-best lineup, which is
    # exactly the lineup the other side's own optimisation starts from.
    best_a = float(a.win_prob[a.best_by_win_prob])
    # Find b's probability for the lineup a assumed it would set.
    assumed = set(a.opp_lineup.tolist())
    match = [i for i in range(len(b.lineups)) if set(b.lineups[i].tolist()) == assumed]
    assert match, "a's assumed opponent lineup should be one of b's legal lineups"
    assert best_a + float(b.win_prob[match[0]]) == pytest.approx(1.0, abs=0.01)


def test_swings_point_the_right_way(db):
    with session(db) as conn:
        advice = optimise_week(conn, "test", 2023, 4, 2, n_sims=40_000, seed=2)
    swings = advice.player_swings()
    assert swings
    for _i, swing, _points, starts in swings:
        if starts:
            assert swing >= -1e-9, "benching an optimal starter cannot help"
        else:
            assert swing <= 1e-9, "forcing in a benched player cannot help"


def test_backtest_scores_the_week_against_real_stat_lines(db):
    with session(db) as conn:
        truth = actual_points(conn, 2023, 4, {"rec": 1.0, "pass_yd": 0.04, "rush_yd": 0.1, "rec_yd": 0.1})
        records = backtest_week(conn, "test", 2023, 4, n_sims=40_000)
    assert truth
    assert len(records) == 2  # four teams, two matchups
    for r in records:
        assert 0.0 <= r.p_home_model <= 1.0
        assert r.home_won in (0.0, 0.5, 1.0)
        assert r.home_actual >= 0 and r.away_actual >= 0
        # The winner of the replay is whoever actually scored more.
        if r.home_actual > r.away_actual:
            assert r.home_won == 1.0


def test_decision_backtest_scores_both_rules_on_the_same_opponent(db):
    with session(db) as conn:
        df = decision_backtest(conn, "test", [2023], [4], n_sims=40_000, progress=False)
    assert len(df) == 4  # one decision per team
    assert (df["won_chosen"].isin([0.0, 0.5, 1.0])).all()
    # Where the rules agree, every column must agree too.
    same = df[~df["differs"]]
    assert (same["actual_chosen"] == same["actual_points"]).all()
    assert (same["won_chosen"] == same["won_points"]).all()
    # The win-probability pick is never worse by its own measure.
    assert (df["p_win_chosen"] >= df["p_win_points"] - 1e-9).all()
