"""Read-only ESPN fantasy football client.

Pulls league settings, rosters and matchups. Public leagues need only a league
id; private ones need the ``espn_s2`` and ``SWID`` cookies from a logged-in
browser session. Nothing here writes to ESPN.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

from ..db import session
from ..model.scoring import espn_scoring_to_rules

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"

# ESPN defaultPositionId -> our position label.
POSITION_IDS: dict[int, str] = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}

# ESPN proTeamId -> nflverse team abbreviation.
PRO_TEAMS: dict[int, str] = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA",
    16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI",
    23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR",
    30: "JAX", 33: "BAL", 34: "HOU",
}

# Injury statuses that mean a zero, not a reduced projection.
OUT_STATUSES: frozenset[str] = frozenset({"OUT", "INJURY_RESERVE", "SUSPENSION", "NOT_ACTIVE", "DOUBTFUL"})


class EspnError(RuntimeError):
    pass


class EspnClient:
    def __init__(self, league_id: str, season: int, espn_s2: str | None = None, swid: str | None = None):
        self.league_id = str(league_id)
        self.season = int(season)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "boombust/0.1"})
        if espn_s2 and swid:
            self.session.cookies.update({"espn_s2": espn_s2, "SWID": swid})

    # --- transport --------------------------------------------------------
    def _get(self, views: Iterable[str], **params: Any) -> dict[str, Any]:
        url = f"{BASE}/seasons/{self.season}/segments/0/leagues/{self.league_id}"
        query: list[tuple[str, Any]] = [("view", v) for v in views]
        query += [(k, v) for k, v in params.items() if v is not None]
        resp = self.session.get(url, params=query, timeout=60)
        if resp.status_code == 401:
            raise EspnError(
                "ESPN returned 401. The league is private: set BOOMBUST_ESPN_S2 and BOOMBUST_SWID in .env"
            )
        if resp.status_code == 404:
            raise EspnError(f"ESPN has no league {self.league_id} for season {self.season}")
        resp.raise_for_status()
        payload = resp.json()
        # Historical seasons come back wrapped in a single-element list.
        if isinstance(payload, list):
            if not payload:
                raise EspnError("ESPN returned an empty payload")
            payload = payload[0]
        return payload

    # --- reads ------------------------------------------------------------
    def settings(self) -> dict[str, Any]:
        return self._get(["mSettings"])

    def rosters(self, week: int | None = None) -> dict[str, Any]:
        return self._get(["mRoster", "mTeam"], scoringPeriodId=week)

    def matchups(self, week: int | None = None) -> dict[str, Any]:
        return self._get(["mMatchupScore", "mTeam"], scoringPeriodId=week)


# --- payload -> rows ------------------------------------------------------
def parse_settings(payload: Mapping[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {}) or {}
    scoring = settings.get("scoringSettings", {}) or {}
    roster = settings.get("rosterSettings", {}) or {}
    rules, unmapped = espn_scoring_to_rules(scoring.get("scoringItems", []) or [])
    if unmapped:
        print(f"  note: {len(unmapped)} ESPN stat ids are not modelled and score 0: {unmapped}")
    slots = {int(k): int(v) for k, v in (roster.get("lineupSlotCounts", {}) or {}).items() if int(v) > 0}
    return {
        "name": settings.get("name") or f"League {payload.get('id')}",
        "current_week": payload.get("scoringPeriodId") or payload.get("status", {}).get("latestScoringPeriod"),
        "scoring": rules,
        "slots": slots,
    }


def parse_teams(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for team in payload.get("teams", []) or []:
        name = team.get("name") or " ".join(
            x for x in (team.get("location"), team.get("nickname")) if x
        ) or f"Team {team.get('id')}"
        owners = team.get("owners") or []
        out.append({"team_id": int(team["id"]), "name": name.strip(), "owner": owners[0] if owners else None})
    return out


def parse_rosters(payload: Mapping[str, Any], week: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team in payload.get("teams", []) or []:
        team_id = int(team["id"])
        entries = ((team.get("roster") or {}).get("entries")) or []
        for entry in entries:
            player = ((entry.get("playerPoolEntry") or {}).get("player")) or entry.get("player") or {}
            pos_id = int(player.get("defaultPositionId", 0) or 0)
            rows.append(
                {
                    "week": week,
                    "team_id": team_id,
                    "espn_player_id": str(player.get("id") or entry.get("playerId") or ""),
                    "name": player.get("fullName") or "",
                    "position": POSITION_IDS.get(pos_id, f"POS{pos_id}"),
                    "pro_team": PRO_TEAMS.get(int(player.get("proTeamId", 0) or 0), "FA"),
                    "eligible_slots": json.dumps(sorted(int(s) for s in (player.get("eligibleSlots") or []))),
                    "lineup_slot_id": entry.get("lineupSlotId"),
                    "injury_status": (player.get("injuryStatus") or "ACTIVE"),
                }
            )
    return rows


def parse_matchups(payload: Mapping[str, Any], week: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for m in payload.get("schedule", []) or []:
        wk = int(m.get("matchupPeriodId") or 0)
        if week is not None and wk != week:
            continue
        home, away = m.get("home") or {}, m.get("away") or {}
        if not home or not away:
            continue  # bye week in an odd-sized league
        rows.append(
            {
                "week": wk,
                "home_team_id": int(home.get("teamId")),
                "away_team_id": int(away.get("teamId")),
                "home_points": home.get("totalPoints"),
                "away_points": away.get("totalPoints"),
                "home_projected": (home.get("totalProjectedPointsLive") or home.get("totalProjectedPoints")),
                "away_projected": (away.get("totalProjectedPointsLive") or away.get("totalProjectedPoints")),
            }
        )
    return rows


# --- sync -----------------------------------------------------------------
def sync_league(
    client: EspnClient,
    weeks: Iterable[int],
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Pull settings, teams, rosters and matchups for the given weeks."""
    weeks = list(weeks)
    meta = parse_settings(client.settings())
    teams = parse_teams(client.rosters())
    counts = {"teams": 0, "roster_slots": 0, "matchups": 0}
    with session(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO leagues (league_id, season, name, current_week, scoring_json, slots_json, source)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                client.league_id,
                client.season,
                meta["name"],
                meta["current_week"],
                json.dumps(meta["scoring"]),
                json.dumps({str(k): v for k, v in meta["slots"].items()}),
                "espn",
            ),
        )
        for t in teams:
            conn.execute(
                "INSERT OR REPLACE INTO fantasy_teams (league_id, season, team_id, name, owner) VALUES (?,?,?,?,?)",
                (client.league_id, client.season, t["team_id"], t["name"], t["owner"]),
            )
            counts["teams"] += 1
        for week in weeks:
            for r in parse_rosters(client.rosters(week), week):
                conn.execute(
                    "INSERT OR REPLACE INTO roster_slots (league_id, season, week, team_id, espn_player_id, name,"
                    " position, pro_team, eligible_slots, lineup_slot_id, injury_status)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        client.league_id, client.season, r["week"], r["team_id"], r["espn_player_id"],
                        r["name"], r["position"], r["pro_team"], r["eligible_slots"],
                        r["lineup_slot_id"], r["injury_status"],
                    ),
                )
                counts["roster_slots"] += 1
        for m in parse_matchups(client.matchups()):
            if weeks and m["week"] not in weeks:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO matchups (league_id, season, week, home_team_id, away_team_id,"
                " home_points, away_points, home_projected, away_projected) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    client.league_id, client.season, m["week"], m["home_team_id"], m["away_team_id"],
                    m["home_points"], m["away_points"], m["home_projected"], m["away_projected"],
                ),
            )
            counts["matchups"] += 1
    return counts
