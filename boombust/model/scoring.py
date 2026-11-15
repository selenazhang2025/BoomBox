"""League scoring rules and the lineup-slot taxonomy.

Raw stat lines live in SQLite; points are a function of (stat line, rules), so a
single ingest serves standard, half-PPR, PPR or any custom ESPN league.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

# --- canonical stat names -------------------------------------------------
# These are the keys a scoring ruleset is written in. Keeping our own names
# (instead of ESPN stat ids or nflverse column names) means the mapping work
# happens once, at the edges.
OFFENSE_STATS: tuple[str, ...] = (
    "pass_yd",
    "pass_td",
    "pass_int",
    "pass_2pt",
    "rush_yd",
    "rush_td",
    "rush_2pt",
    "rec",
    "rec_yd",
    "rec_td",
    "rec_2pt",
    "fumble_lost",
    "ret_td",
)
KICKER_STATS: tuple[str, ...] = (
    "fg_0_39",
    "fg_40_49",
    "fg_50",
    "fg_miss",
    "pat_made",
    "pat_miss",
)
DST_STATS: tuple[str, ...] = (
    "dst_sack",
    "dst_int",
    "dst_fum_rec",
    "dst_safety",
    "dst_td",
)
# Points-allowed buckets are mutually exclusive indicators (exactly one fires).
DST_PA_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("dst_pa_0", 0, 0),
    ("dst_pa_1_6", 1, 6),
    ("dst_pa_7_13", 7, 13),
    ("dst_pa_14_17", 14, 17),
    ("dst_pa_18_21", 18, 21),
    ("dst_pa_22_27", 22, 27),
    ("dst_pa_28_34", 28, 34),
    ("dst_pa_35_45", 35, 45),
    ("dst_pa_46", 46, 999),
)
ALL_STATS: tuple[str, ...] = (
    OFFENSE_STATS + KICKER_STATS + DST_STATS + tuple(b[0] for b in DST_PA_BUCKETS)
)

# --- default ruleset ------------------------------------------------------
# ESPN's standard PPR defaults. Used for synthetic leagues and as a fallback
# for any stat a real league does not configure.
PPR_RULES: dict[str, float] = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "pass_2pt": 2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rush_2pt": 2.0,
    "rec": 1.0,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "rec_2pt": 2.0,
    "fumble_lost": -2.0,
    "ret_td": 6.0,
    "fg_0_39": 3.0,
    "fg_40_49": 4.0,
    "fg_50": 5.0,
    "fg_miss": -1.0,
    "pat_made": 1.0,
    "pat_miss": -1.0,
    "dst_sack": 1.0,
    "dst_int": 2.0,
    "dst_fum_rec": 2.0,
    "dst_safety": 2.0,
    "dst_td": 6.0,
    "dst_pa_0": 5.0,
    "dst_pa_1_6": 4.0,
    "dst_pa_7_13": 3.0,
    "dst_pa_14_17": 1.0,
    "dst_pa_18_21": 0.0,
    "dst_pa_22_27": -1.0,
    "dst_pa_28_34": -3.0,
    "dst_pa_35_45": -5.0,
    "dst_pa_46": -5.0,
}

STANDARD_RULES: dict[str, float] = {**PPR_RULES, "rec": 0.0}
HALF_PPR_RULES: dict[str, float] = {**PPR_RULES, "rec": 0.5}

# --- ESPN stat id -> canonical name --------------------------------------
# ESPN exposes league scoring as a list of {statId, points}. Only the ids that
# map onto a stat we actually store are listed; anything else is reported as
# unmapped rather than silently dropped.
ESPN_STAT_IDS: dict[int, str] = {
    3: "pass_yd",
    4: "pass_td",
    19: "pass_2pt",
    20: "pass_int",
    24: "rush_yd",
    25: "rush_td",
    26: "rush_2pt",
    42: "rec_yd",
    43: "rec_td",
    44: "rec_2pt",
    53: "rec",
    72: "fumble_lost",
    74: "fg_50",
    77: "fg_40_49",
    80: "fg_0_39",
    85: "fg_miss",
    86: "pat_made",
    88: "pat_miss",
    89: "dst_pa_0",
    90: "dst_pa_1_6",
    91: "dst_pa_7_13",
    92: "dst_pa_14_17",
    93: "dst_td",
    95: "dst_int",
    96: "dst_fum_rec",
    98: "dst_safety",
    99: "dst_sack",
    101: "ret_td",
    102: "ret_td",
    103: "dst_td",
    104: "dst_td",
    120: "dst_pa_18_21",
    121: "dst_pa_22_27",
    122: "dst_pa_28_34",
    123: "dst_pa_35_45",
    124: "dst_pa_46",
}

# --- lineup slots ---------------------------------------------------------
# ESPN lineupSlotId -> (label, positions eligible for the slot).
SLOT_LABELS: dict[int, str] = {
    0: "QB",
    2: "RB",
    3: "RB/WR",
    4: "WR",
    5: "WR/TE",
    6: "TE",
    7: "OP",
    16: "D/ST",
    17: "K",
    20: "BE",
    21: "IR",
    23: "FLEX",
}
SLOT_POSITIONS: dict[int, frozenset[str]] = {
    0: frozenset({"QB"}),
    2: frozenset({"RB"}),
    3: frozenset({"RB", "WR"}),
    4: frozenset({"WR"}),
    5: frozenset({"WR", "TE"}),
    6: frozenset({"TE"}),
    7: frozenset({"QB", "RB", "WR", "TE"}),
    16: frozenset({"D/ST", "DST"}),
    17: frozenset({"K"}),
    23: frozenset({"RB", "WR", "TE"}),
}
BENCH_SLOTS: frozenset[int] = frozenset({20, 21})

# A conventional 10-starter league, used for synthetic leagues.
DEFAULT_SLOT_COUNTS: dict[int, int] = {0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1, 20: 7}


def starting_slots(slot_counts: Mapping[int | str, int]) -> list[int]:
    """Expand a {slot id: count} mapping into the ordered list of starting slots.

    Bench and IR slots are dropped. Slots are ordered by how restrictive they
    are (fewest eligible positions first) which makes lineup enumeration prune
    earlier and keeps output in a readable QB-first order.
    """
    expanded: list[int] = []
    for raw_slot, count in slot_counts.items():
        slot = int(raw_slot)
        if slot in BENCH_SLOTS or slot not in SLOT_POSITIONS:
            continue
        expanded.extend([slot] * int(count))
    return sorted(expanded, key=lambda s: (len(SLOT_POSITIONS[s]), s))


def espn_scoring_to_rules(scoring_items: list[Mapping[str, Any]]) -> tuple[dict[str, float], list[int]]:
    """Translate ESPN ``scoringItems`` into a canonical ruleset.

    Returns the rules plus the stat ids we could not map, so a surprising league
    setup shows up as a warning instead of quietly scoring zero.
    """
    rules = dict(PPR_RULES)
    seen: set[str] = set()
    unmapped: list[int] = []
    for item in scoring_items:
        stat_id = int(item.get("statId", -1))
        name = ESPN_STAT_IDS.get(stat_id)
        if name is None:
            unmapped.append(stat_id)
            continue
        points = float(item.get("points", 0.0) or 0.0)
        # Several ESPN ids collapse onto one canonical stat (three distinct
        # return-touchdown ids, for instance); take the first and keep going.
        if name in seen:
            continue
        seen.add(name)
        rules[name] = points
    return rules, sorted(set(unmapped))


# --- nflverse stat line -> canonical stats -------------------------------
def offense_stats_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map nflverse weekly player columns onto canonical offensive stats."""

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    out = pd.DataFrame(index=df.index)
    out["pass_yd"] = col("passing_yards")
    out["pass_td"] = col("passing_tds")
    out["pass_int"] = col("passing_interceptions")
    out["pass_2pt"] = col("passing_2pt_conversions")
    out["rush_yd"] = col("rushing_yards")
    out["rush_td"] = col("rushing_tds")
    out["rush_2pt"] = col("rushing_2pt_conversions")
    out["rec"] = col("receptions")
    out["rec_yd"] = col("receiving_yards")
    out["rec_td"] = col("receiving_tds")
    out["rec_2pt"] = col("receiving_2pt_conversions")
    out["fumble_lost"] = (
        col("rushing_fumbles_lost") + col("receiving_fumbles_lost") + col("sack_fumbles_lost")
    )
    out["ret_td"] = col("special_teams_tds")
    out["fg_0_39"] = col("fg_made_0_19") + col("fg_made_20_29") + col("fg_made_30_39")
    out["fg_40_49"] = col("fg_made_40_49")
    out["fg_50"] = col("fg_made_50_59") + col("fg_made_60_")
    out["fg_miss"] = col("fg_missed")
    out["pat_made"] = col("pat_made")
    out["pat_miss"] = col("pat_missed")
    for name in DST_STATS:
        out[name] = 0.0
    for name, _lo, _hi in DST_PA_BUCKETS:
        out[name] = 0.0
    return out


def dst_stats_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map nflverse weekly *team* defensive columns onto canonical D/ST stats."""

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    out = pd.DataFrame(index=df.index)
    for name in OFFENSE_STATS + KICKER_STATS:
        out[name] = 0.0
    out["dst_sack"] = col("def_sacks")
    out["dst_int"] = col("def_interceptions")
    out["dst_fum_rec"] = col("fumble_recovery_opp")
    out["dst_safety"] = col("def_safeties")
    out["dst_td"] = col("def_tds") + col("fumble_recovery_tds")
    pa = col("points_allowed")
    for name, lo, hi in DST_PA_BUCKETS:
        out[name] = ((pa >= lo) & (pa <= hi)).astype("float64")
    return out


def apply_rules(stats: pd.DataFrame, rules: Mapping[str, float]) -> pd.Series:
    """Dot the canonical stat frame with a ruleset to get fantasy points."""
    total = pd.Series(0.0, index=stats.index)
    for name in ALL_STATS:
        weight = float(rules.get(name, 0.0))
        if weight == 0.0 or name not in stats.columns:
            continue
        total = total + stats[name].astype("float64") * weight
    return total
