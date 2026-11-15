"""Scoring rules: stat lines in, fantasy points out."""

from __future__ import annotations

import pandas as pd
import pytest

from boombust.model.scoring import (
    DEFAULT_SLOT_COUNTS,
    HALF_PPR_RULES,
    PPR_RULES,
    STANDARD_RULES,
    apply_rules,
    dst_stats_frame,
    espn_scoring_to_rules,
    offense_stats_frame,
    starting_slots,
)


def test_a_receiving_line_scores_the_textbook_amount():
    line = pd.DataFrame([{"receptions": 8, "receiving_yards": 115, "receiving_tds": 1}])
    stats = offense_stats_frame(line)
    # 8 catches + 11.5 yards + 6 for the score
    assert apply_rules(stats, PPR_RULES).iloc[0] == pytest.approx(8 + 11.5 + 6)
    assert apply_rules(stats, HALF_PPR_RULES).iloc[0] == pytest.approx(4 + 11.5 + 6)
    assert apply_rules(stats, STANDARD_RULES).iloc[0] == pytest.approx(11.5 + 6)


def test_a_quarterback_line_includes_the_interception_penalty():
    line = pd.DataFrame(
        [{"passing_yards": 300, "passing_tds": 2, "passing_interceptions": 1, "rushing_yards": 20}]
    )
    points = apply_rules(offense_stats_frame(line), PPR_RULES).iloc[0]
    assert points == pytest.approx(12 + 8 - 2 + 2)


def test_fumbles_from_every_source_are_counted_once_each():
    line = pd.DataFrame(
        [{"rushing_fumbles_lost": 1, "receiving_fumbles_lost": 1, "sack_fumbles_lost": 1}]
    )
    assert apply_rules(offense_stats_frame(line), PPR_RULES).iloc[0] == pytest.approx(-6)


def test_field_goals_are_bucketed_by_distance():
    line = pd.DataFrame(
        [{"fg_made_30_39": 1, "fg_made_40_49": 1, "fg_made_50_59": 1, "fg_made_60_": 1, "pat_made": 3}]
    )
    # 3 + 4 + 5 + 5 for the kicks, 3 for the extra points
    assert apply_rules(offense_stats_frame(line), PPR_RULES).iloc[0] == pytest.approx(20)


def test_points_allowed_fires_exactly_one_bucket():
    teams = pd.DataFrame([{"def_sacks": 4, "def_interceptions": 2, "points_allowed": 10}])
    stats = dst_stats_frame(teams)
    buckets = [c for c in stats.columns if c.startswith("dst_pa_")]
    assert stats[buckets].iloc[0].sum() == 1
    assert stats["dst_pa_7_13"].iloc[0] == 1
    # 4 sacks + 2 picks + the 7-13 bucket
    assert apply_rules(stats, PPR_RULES).iloc[0] == pytest.approx(4 + 4 + 3)


def test_a_shutout_takes_the_top_bucket():
    teams = pd.DataFrame([{"points_allowed": 0}])
    assert dst_stats_frame(teams)["dst_pa_0"].iloc[0] == 1


def test_offense_and_defence_frames_share_a_column_set():
    """apply_rules dots a single vector of rules against either frame, so the
    two have to agree on their columns."""
    offense = offense_stats_frame(pd.DataFrame([{"receptions": 1}]))
    defence = dst_stats_frame(pd.DataFrame([{"def_sacks": 1, "points_allowed": 20}]))
    assert set(offense.columns) == set(defence.columns)


# --- ESPN translation -----------------------------------------------------
def test_espn_scoring_overrides_defaults_and_reports_the_rest():
    rules, unmapped = espn_scoring_to_rules(
        [{"statId": 53, "points": 0.5}, {"statId": 42, "points": 0.1}, {"statId": 4242, "points": 9}]
    )
    assert rules["rec"] == 0.5
    assert rules["rec_yd"] == 0.1
    assert rules["pass_td"] == PPR_RULES["pass_td"]  # untouched keys keep the default
    assert unmapped == [4242]


def test_duplicate_espn_ids_for_one_stat_take_the_first():
    """Three distinct ESPN ids all mean 'defensive touchdown'."""
    rules, _ = espn_scoring_to_rules([{"statId": 93, "points": 6}, {"statId": 103, "points": 99}])
    assert rules["dst_td"] == 6


# --- slots ----------------------------------------------------------------
def test_starting_slots_drops_the_bench_and_orders_by_restrictiveness():
    slots = starting_slots(DEFAULT_SLOT_COUNTS)
    assert 20 not in slots and 21 not in slots
    assert len(slots) == 9
    assert slots[0] == 0           # QB: single eligible position, so it comes first
    assert slots[-1] == 23         # FLEX: the loosest slot, last
    assert slots.count(2) == 2 and slots.count(4) == 2


def test_unknown_slot_ids_are_ignored_rather_than_crashing():
    assert starting_slots({0: 1, 999: 3}) == [0]
