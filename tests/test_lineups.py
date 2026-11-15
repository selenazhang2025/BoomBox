"""Lineup enumeration, cross-checked against brute force.

The fast enumerator prunes interchangeable slots and dedups by starter set, so
it is worth proving it agrees with an exhaustive "try every subset" search.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from boombust.model.scoring import SLOT_POSITIONS, starting_slots
from boombust.sim.lineups import (
    LineupInfeasible,
    RosterSpec,
    assign_slots,
    enumerate_lineups,
    enumerate_with_trim,
)


def make_spec(positions: list[str]) -> RosterSpec:
    eligibility = [
        frozenset({s for s, allowed in SLOT_POSITIONS.items() if pos in allowed}) for pos in positions
    ]
    return RosterSpec(names=[f"{p}{i}" for i, p in enumerate(positions)], positions=positions, eligibility=eligibility)


def brute_force(spec: RosterSpec, slots: list[int]) -> set[tuple[int, ...]]:
    """Every subset of the right size that admits a full slot assignment."""
    out = set()
    for subset in combinations(range(len(spec)), len(slots)):
        if len(assign_slots(spec, subset, slots)) == len(slots):
            out.add(tuple(sorted(subset)))
    return out


ROSTER = ["QB", "QB", "RB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "TE", "K", "D/ST"]


def test_matches_brute_force():
    spec = make_spec(ROSTER)
    slots = starting_slots({0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1, 20: 5})
    fast = {tuple(row) for row in enumerate_lineups(spec, slots)}
    assert fast == brute_force(spec, slots)
    assert len(fast) > 100  # the test would be vacuous on a tiny search space


def test_every_lineup_is_assignable_and_unique():
    spec = make_spec(ROSTER)
    slots = starting_slots({0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1, 20: 5})
    lineups = enumerate_lineups(spec, slots)
    assert lineups.shape[1] == len(slots)
    seen = set()
    for row in lineups:
        key = tuple(sorted(row.tolist()))
        assert key not in seen, "duplicate starter set"
        seen.add(key)
        assert len(set(row.tolist())) == len(row), "a player started twice"
        assert len(assign_slots(spec, row.tolist(), slots)) == len(slots)


def test_flex_lets_a_te_start_over_a_wr():
    spec = make_spec(["QB", "RB", "RB", "WR", "WR", "TE", "TE", "K", "D/ST"])
    slots = starting_slots({0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1})
    lineups = enumerate_lineups(spec, slots)
    # Only nine players for nine slots, so there is exactly one legal lineup
    # and it must use the second TE in the flex.
    assert lineups.shape == (1, 9)
    assert sorted(lineups[0].tolist()) == list(range(9))


def test_missing_required_position_raises():
    spec = make_spec(["QB", "RB", "RB", "WR", "WR", "TE", "D/ST"])  # no kicker
    slots = starting_slots({0: 1, 2: 2, 4: 2, 6: 1, 17: 1, 16: 1})
    with pytest.raises(LineupInfeasible):
        enumerate_lineups(spec, slots)


def test_trim_drops_lowest_priority_players_until_under_cap():
    spec = make_spec(ROSTER + ["RB", "WR", "WR"])
    slots = starting_slots({0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1})
    priority = list(np.arange(len(spec), dtype=float))  # player 0 is worst
    lineups, dropped = enumerate_with_trim(spec, slots, priority, max_lineups=500)
    assert len(lineups) <= 500
    assert dropped, "expected the cap to force a trim"
    assert dropped[0] == 0  # lowest priority went first
    assert all(d not in set(lineups.ravel().tolist()) for d in dropped)
