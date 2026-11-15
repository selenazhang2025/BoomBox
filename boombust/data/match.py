"""Match ESPN roster entries to nflverse player ids.

ESPN and nflverse disagree about suffixes, punctuation and the occasional
nickname, so matching is name-normalised and position-guarded, with the pro
team as a tiebreak.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z ]", " ", text).lower()
    parts = [p for p in text.split() if p and p not in SUFFIXES]
    return " ".join(parts)


def build_index(conn: sqlite3.Connection, season: int) -> dict[tuple[str, str], list[sqlite3.Row]]:
    """Index nflverse players who appeared in ``season`` by (name, position)."""
    rows = conn.execute(
        "SELECT DISTINCT p.player_id, p.name, pw.position, pw.team"
        " FROM players p JOIN player_weeks pw ON pw.player_id = p.player_id"
        " WHERE pw.season = ?",
        (season,),
    ).fetchall()
    index: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        index.setdefault((normalise(row["name"]), row["position"] or ""), []).append(row)
    return index


# ESPN lumps several NFL positions into one fantasy slot.
POSITION_ALIASES: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "RB": ("RB", "FB"),
    "WR": ("WR",),
    "TE": ("TE",),
    "K": ("K",),
}


def match_one(
    index: dict[tuple[str, str], list[sqlite3.Row]],
    name: str,
    position: str,
    pro_team: str | None = None,
) -> str | None:
    key_name = normalise(name)
    for pos in POSITION_ALIASES.get(position, (position,)):
        candidates = index.get((key_name, pos))
        if not candidates:
            continue
        if len(candidates) == 1:
            return candidates[0]["player_id"]
        if pro_team:
            for cand in candidates:
                if cand["team"] == pro_team:
                    return cand["player_id"]
        return candidates[0]["player_id"]
    return None


def link_roster_ids(conn: sqlite3.Connection, league_id: str, season: int) -> tuple[int, list[str]]:
    """Fill ``roster_slots.player_id`` for a synced league. Returns (matched, unmatched names)."""
    index = build_index(conn, season)
    rows = conn.execute(
        "SELECT rowid, name, position, pro_team FROM roster_slots"
        " WHERE league_id = ? AND season = ? AND player_id IS NULL",
        (league_id, season),
    ).fetchall()
    matched, unmatched = 0, []
    for row in rows:
        if row["position"] == "D/ST":
            # Team defences are not nflverse players; key them by team instead.
            conn.execute(
                "UPDATE roster_slots SET player_id = ? WHERE rowid = ?",
                (f"DST_{row['pro_team']}", row["rowid"]),
            )
            matched += 1
            continue
        pid = match_one(index, row["name"], row["position"], row["pro_team"])
        if pid:
            conn.execute("UPDATE roster_slots SET player_id = ? WHERE rowid = ?", (pid, row["rowid"]))
            matched += 1
        else:
            unmatched.append(f"{row['name']} ({row['position']})")
    return matched, unmatched
