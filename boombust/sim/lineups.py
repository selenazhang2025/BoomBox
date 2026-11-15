"""Enumerate every legal starting lineup.

A lineup's score depends only on *who* starts, not on which slot each player
fills, so the unit of enumeration is the set of starters that admits at least
one valid slot assignment. Two assignments of the same nine players are one
lineup, and are counted once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..model.scoring import SLOT_LABELS, SLOT_POSITIONS


@dataclass
class RosterSpec:
    """The enumeration inputs: who is on the roster and where they can play."""

    names: list[str]
    positions: list[str]
    eligibility: list[frozenset[int]]  # slot ids each player can fill

    def __len__(self) -> int:
        return len(self.names)


def eligibility_from_rows(rows: Sequence[Mapping[str, object]]) -> RosterSpec:
    """Read eligibility off roster rows, falling back to position defaults."""
    names, positions, eligibility = [], [], []
    for row in rows:
        position = str(row.get("position") or "")
        if position == "FB":
            position = "RB"
        raw = row.get("eligible_slots")
        slots: set[int] = set()
        if isinstance(raw, str) and raw:
            try:
                slots = {int(s) for s in json.loads(raw)}
            except (ValueError, TypeError):
                slots = set()
        elif isinstance(raw, (list, tuple)):
            slots = {int(s) for s in raw}
        if not slots:
            slots = {s for s, pos in SLOT_POSITIONS.items() if position in pos}
        names.append(str(row.get("name") or row.get("player_id") or "?"))
        positions.append(position)
        eligibility.append(frozenset(slots))
    return RosterSpec(names=names, positions=positions, eligibility=eligibility)


def _candidates(spec: RosterSpec, slots: Sequence[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for slot in set(slots):
        allowed = SLOT_POSITIONS.get(slot, frozenset())
        out[slot] = [
            i
            for i in range(len(spec))
            # Trust the league's own eligibility list, but require the position
            # to make sense too: ESPN hands out slot ids we do not model.
            if slot in spec.eligibility[i] and (not allowed or spec.positions[i] in allowed)
        ]
    return out


class LineupOverflow(RuntimeError):
    """Raised when a roster admits more lineups than the caller allowed."""


class LineupInfeasible(LineupOverflow):
    """Raised when the player pool cannot fill every required slot."""


def enumerate_lineups(
    spec: RosterSpec,
    slots: Sequence[int],
    *,
    max_lineups: int = 200_000,
    restrict_to: Sequence[int] | None = None,
) -> np.ndarray:
    """All legal lineups as an (L, K) array of roster indices.

    ``restrict_to`` limits the player pool, which is how the caller trims a
    roster that would otherwise produce an unmanageable number of lineups.
    """
    slots = list(slots)
    k = len(slots)
    if k == 0:
        return np.zeros((0, 0), dtype=np.int32)
    cands = _candidates(spec, slots)
    if restrict_to is not None:
        keep = set(int(i) for i in restrict_to)
        cands = {slot: [i for i in players if i in keep] for slot, players in cands.items()}
    for slot in slots:
        if not cands.get(slot):
            raise LineupInfeasible(
                f"no eligible player for a required {SLOT_LABELS.get(slot, slot)} slot"
            )

    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    current = [-1] * k

    def walk(depth: int, used: int) -> None:
        if len(out) >= max_lineups:
            raise LineupOverflow(
                f"roster admits more than {max_lineups:,} lineups; trim the pool or raise the cap"
            )
        if depth == k:
            key = tuple(sorted(current))
            if key not in seen:
                seen.add(key)
                out.append(key)
            return
        slot = slots[depth]
        # Identical adjacent slots (two RB, two WR) are interchangeable, so
        # force increasing player index across the run and skip the permutations.
        floor = current[depth - 1] + 1 if depth and slots[depth - 1] == slot else 0
        for player in cands[slot]:
            if player < floor:
                continue
            bit = 1 << player
            if used & bit:
                continue
            current[depth] = player
            walk(depth + 1, used | bit)
        current[depth] = -1

    walk(0, 0)
    return np.array(sorted(out), dtype=np.int32) if out else np.zeros((0, k), dtype=np.int32)


def count_lineups(spec: RosterSpec, slots: Sequence[int], *, max_lineups: int = 10_000_000) -> int:
    return len(enumerate_lineups(spec, slots, max_lineups=max_lineups))


def enumerate_with_trim(
    spec: RosterSpec,
    slots: Sequence[int],
    priority: Sequence[float],
    *,
    max_lineups: int = 200_000,
) -> tuple[np.ndarray, list[int]]:
    """Enumerate lineups, dropping the least promising bench players if needed.

    ``priority`` is one score per player (projected mean works well). Players
    are dropped lowest-first until the lineup count fits under the cap. Returns
    the lineups and the indices of any players that were excluded.
    """
    order = sorted(range(len(spec)), key=lambda i: priority[i])
    dropped: list[int] = []
    pool = list(range(len(spec)))
    while True:
        try:
            lineups = enumerate_lineups(spec, slots, max_lineups=max_lineups, restrict_to=pool)
            return lineups, dropped
        except LineupInfeasible:
            # The pool itself is short of a required position; trimming cannot help.
            raise
        except LineupOverflow:
            # Too many lineups. Drop the least promising player whose removal
            # still leaves every slot fillable, and try again.
            victim = next(
                (i for i in order if i in pool and is_feasible(spec, slots, [p for p in pool if p != i])),
                None,
            )
            if victim is None:
                raise
            pool.remove(victim)
            dropped.append(victim)


def is_feasible(spec: RosterSpec, slots: Sequence[int], pool: Sequence[int]) -> bool:
    """Can this player pool fill every starting slot at once?"""
    return len(assign_slots(spec, list(pool), slots)) == len(slots)


# --- slot assignment, for display ----------------------------------------
def assign_slots(spec: RosterSpec, lineup: Sequence[int], slots: Sequence[int]) -> list[tuple[int, int]]:
    """Find one valid (slot, player) assignment for a lineup, via augmenting paths."""
    slots = list(slots)
    match_slot: dict[int, int] = {}

    def try_assign(player: int, seen: set[int]) -> bool:
        for si, slot in enumerate(slots):
            if si in seen:
                continue
            allowed = SLOT_POSITIONS.get(slot, frozenset())
            if slot not in spec.eligibility[player]:
                continue
            if allowed and spec.positions[player] not in allowed:
                continue
            seen.add(si)
            if si not in match_slot or try_assign(match_slot[si], seen):
                match_slot[si] = player
                return True
        return False

    for player in lineup:
        try_assign(player, set())
    return [(slots[si], player) for si, player in sorted(match_slot.items())]
