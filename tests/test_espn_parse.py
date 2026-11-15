"""ESPN payload parsing, checked against hand-built fixtures.

The real API needs credentials and a live league, so the parsers are tested
against the shapes ESPN actually returns rather than over the network.
"""

from __future__ import annotations

import json

from boombust.data.espn import parse_matchups, parse_rosters, parse_settings, parse_teams

SETTINGS_PAYLOAD = {
    "id": 42,
    "scoringPeriodId": 7,
    "settings": {
        "name": "Test League",
        "scoringSettings": {"scoringItems": [{"statId": 53, "points": 0.5}, {"statId": 4, "points": 6.0}]},
        "rosterSettings": {
            "lineupSlotCounts": {"0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "16": 1, "17": 1, "20": 7, "21": 1}
        },
    },
}

ROSTER_PAYLOAD = {
    "teams": [
        {
            "id": 3,
            "location": "Boom",
            "nickname": "Bust",
            "owners": ["{ABC}"],
            "roster": {
                "entries": [
                    {
                        "lineupSlotId": 2,
                        "playerPoolEntry": {
                            "player": {
                                "id": 999,
                                "fullName": "Alpha Back",
                                "defaultPositionId": 2,
                                "proTeamId": 12,
                                "eligibleSlots": [2, 3, 23, 20],
                                "injuryStatus": "ACTIVE",
                            }
                        },
                    },
                    {
                        "lineupSlotId": 20,
                        "playerPoolEntry": {
                            "player": {
                                "id": 1000,
                                "fullName": "Beta Wide",
                                "defaultPositionId": 3,
                                "proTeamId": 14,
                                "eligibleSlots": [3, 4, 5, 23, 20],
                                "injuryStatus": "QUESTIONABLE",
                            }
                        },
                    },
                ]
            },
        }
    ]
}

MATCHUP_PAYLOAD = {
    "schedule": [
        {
            "matchupPeriodId": 5,
            "home": {"teamId": 1, "totalPoints": 100.5, "totalProjectedPoints": 98.0},
            "away": {"teamId": 2, "totalPoints": 90.0, "totalProjectedPoints": 95.0},
        },
        {"matchupPeriodId": 6, "home": {"teamId": 1}},  # bye: no opponent
    ]
}


def test_parse_settings_maps_scoring_and_slots():
    meta = parse_settings(SETTINGS_PAYLOAD)
    assert meta["name"] == "Test League"
    assert meta["current_week"] == 7
    assert meta["scoring"]["rec"] == 0.5  # half-PPR override took effect
    assert meta["scoring"]["pass_td"] == 6.0
    assert meta["slots"][0] == 1 and meta["slots"][23] == 1
    assert 20 in meta["slots"]  # bench is kept here; dropped by starting_slots()


def test_parse_teams_builds_display_name():
    teams = parse_teams(ROSTER_PAYLOAD)
    assert teams == [{"team_id": 3, "name": "Boom Bust", "owner": "{ABC}"}]


def test_parse_rosters_extracts_eligibility_and_slot():
    rows = parse_rosters(ROSTER_PAYLOAD, week=5)
    assert len(rows) == 2
    rb = rows[0]
    assert rb["name"] == "Alpha Back"
    assert rb["position"] == "RB"
    assert rb["pro_team"] == "KC"
    assert json.loads(rb["eligible_slots"]) == [2, 3, 20, 23]
    assert rb["lineup_slot_id"] == 2
    assert rows[1]["pro_team"] == "LA"
    assert rows[1]["injury_status"] == "QUESTIONABLE"


def test_parse_matchups_skips_byes():
    rows = parse_matchups(MATCHUP_PAYLOAD)
    assert len(rows) == 1
    assert rows[0]["home_points"] == 100.5
    assert rows[0]["away_projected"] == 95.0
