"""SQLite storage: one file holding nflverse history and ESPN league state."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import DEFAULT_DB

# Raw stat columns pulled from nflverse and kept per player-week. Scoring rules
# are applied on top of these at read time, so one ingest serves any league.
PLAYER_STAT_COLUMNS: tuple[str, ...] = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_2pt_conversions",
    "sack_fumbles_lost",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles_lost",
    "receiving_2pt_conversions",
    "special_teams_tds",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "fg_missed",
    "fg_blocked",
    "pat_made",
    "pat_missed",
)

TEAM_STAT_COLUMNS: tuple[str, ...] = (
    "def_sacks",
    "def_interceptions",
    "def_safeties",
    "def_tds",
    "fumble_recovery_opp",
    "fumble_recovery_tds",
)

SCHEMA = f"""
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS players (
    player_id       TEXT PRIMARY KEY,
    name            TEXT,
    position        TEXT,
    last_team       TEXT,
    last_season     INTEGER
);

CREATE TABLE IF NOT EXISTS player_weeks (
    player_id       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    season_type     TEXT,
    team            TEXT,
    opponent        TEXT,
    position        TEXT,
    {",\n    ".join(f"{c} REAL DEFAULT 0" for c in PLAYER_STAT_COLUMNS)},
    PRIMARY KEY (player_id, season, week, season_type)
);
CREATE INDEX IF NOT EXISTS idx_pw_season_week ON player_weeks (season, week);
CREATE INDEX IF NOT EXISTS idx_pw_pos ON player_weeks (position, season, week);

CREATE TABLE IF NOT EXISTS team_weeks (
    team            TEXT NOT NULL,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    season_type     TEXT,
    opponent        TEXT,
    points_allowed  REAL,
    {",\n    ".join(f"{c} REAL DEFAULT 0" for c in TEAM_STAT_COLUMNS)},
    PRIMARY KEY (team, season, week, season_type)
);

CREATE TABLE IF NOT EXISTS games (
    game_id         TEXT PRIMARY KEY,
    season          INTEGER,
    week            INTEGER,
    game_type       TEXT,
    home_team       TEXT,
    away_team       TEXT,
    home_score      REAL,
    away_score      REAL
);
CREATE INDEX IF NOT EXISTS idx_games_season_week ON games (season, week);

-- ESPN league state -------------------------------------------------------
CREATE TABLE IF NOT EXISTS leagues (
    league_id       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    name            TEXT,
    current_week    INTEGER,
    scoring_json    TEXT,   -- json: canonical stat name -> points per unit
    slots_json      TEXT,   -- json: lineup slot id -> count
    source          TEXT,   -- 'espn' or 'synthetic'
    PRIMARY KEY (league_id, season)
);

CREATE TABLE IF NOT EXISTS fantasy_teams (
    league_id       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    team_id         INTEGER NOT NULL,
    name            TEXT,
    owner           TEXT,
    PRIMARY KEY (league_id, season, team_id)
);

CREATE TABLE IF NOT EXISTS roster_slots (
    league_id       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    team_id         INTEGER NOT NULL,
    espn_player_id  TEXT,
    player_id       TEXT,           -- nflverse gsis id, when matched
    name            TEXT,
    position        TEXT,
    pro_team        TEXT,
    eligible_slots  TEXT,           -- json list of lineup slot ids
    lineup_slot_id  INTEGER,        -- what the manager actually started
    injury_status   TEXT,
    PRIMARY KEY (league_id, season, week, team_id, espn_player_id)
);

CREATE TABLE IF NOT EXISTS matchups (
    league_id       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    home_team_id    INTEGER,
    away_team_id    INTEGER,
    home_points     REAL,
    away_points     REAL,
    home_projected  REAL,
    away_projected  REAL,
    PRIMARY KEY (league_id, season, week, home_team_id, away_team_id)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    source          TEXT NOT NULL,
    key             TEXT NOT NULL,
    rows            INTEGER,
    fetched_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source, key)
);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str | None = None) -> Path:
    """Create the schema if it is not already there. Idempotent."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
    return Path(db_path) if db_path else DEFAULT_DB


@contextmanager
def session(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
