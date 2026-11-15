"""Ingest nflverse weekly data into SQLite.

nflverse publishes one parquet per season as a GitHub release asset. We
discover which seasons exist rather than hardcoding a range, cache the files
locally, and store raw stat lines so scoring rules stay a read-time concern.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from ..config import CACHE_DIR, NFLVERSE_API, NFLVERSE_RELEASE
from ..db import PLAYER_STAT_COLUMNS, TEAM_STAT_COLUMNS, session

PLAYER_STATS_TAG = "player_stats"
SCHEDULES_TAG = "schedules"
PLAYER_ASSET = "stats_player_week_{season}.parquet"
TEAM_ASSET = "stats_team_week_{season}.parquet"
GAMES_ASSET = "games.parquet"

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "boombust/0.1 (+https://github.com/)"})


# --- download helpers -----------------------------------------------------
def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def download(tag: str, asset: str, *, refresh: bool = False) -> Path:
    """Fetch a release asset, using the on-disk cache unless ``refresh``."""
    dest = _cache_path(asset)
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    url = f"{NFLVERSE_RELEASE}/{tag}/{asset}"
    resp = _SESSION.get(url, timeout=120)
    if resp.status_code == 404:
        raise FileNotFoundError(f"nflverse has no asset {asset!r} (tried {url})")
    resp.raise_for_status()
    if not resp.content.startswith(b"PAR1"):
        raise ValueError(f"{asset} did not come back as parquet (got {resp.content[:16]!r})")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest


def available_seasons(*, refresh: bool = False) -> list[int]:
    """Seasons for which nflverse has published weekly player stats."""
    cache = _cache_path("_player_stats_seasons.txt")
    if cache.exists() and not refresh:
        return [int(x) for x in cache.read_text().split() if x]
    resp = _SESSION.get(f"{NFLVERSE_API}/{PLAYER_STATS_TAG}", timeout=60)
    resp.raise_for_status()
    names = [a.get("name", "") for a in resp.json().get("assets", [])]
    seasons = sorted(
        {
            int(m.group(1))
            for name in names
            if (m := re.fullmatch(r"stats_player_week_(\d{4})\.parquet", name))
        }
    )
    if seasons:
        cache.write_text(" ".join(str(s) for s in seasons))
    return seasons


def latest_season(*, refresh: bool = False) -> int:
    seasons = available_seasons(refresh=refresh)
    if not seasons:
        raise RuntimeError("could not determine any available nflverse season")
    return seasons[-1]


def parse_seasons(spec: str | None, *, refresh: bool = False) -> list[int]:
    """Parse ``2020-2024`` / ``2023,2024`` / ``2024`` against what exists."""
    have = available_seasons(refresh=refresh)
    if not spec:
        return have[-4:] if len(have) >= 4 else have
    wanted: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = (int(x) for x in chunk.split("-", 1))
            wanted.update(range(lo, hi + 1))
        else:
            wanted.add(int(chunk))
    missing = sorted(wanted - set(have))
    if missing:
        print(f"  note: nflverse has no weekly stats for {missing}; skipping")
    return sorted(wanted & set(have))


# --- frames ---------------------------------------------------------------
def _numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    return df


def load_games(*, refresh: bool = False) -> pd.DataFrame:
    path = download(SCHEDULES_TAG, GAMES_ASSET, refresh=refresh)
    df = pd.read_parquet(path)
    keep = ["game_id", "season", "week", "game_type", "home_team", "away_team", "home_score", "away_score"]
    return df[keep].copy()


def load_player_weeks(season: int, *, refresh: bool = False) -> pd.DataFrame:
    path = download(PLAYER_STATS_TAG, PLAYER_ASSET.format(season=season), refresh=refresh)
    df = pd.read_parquet(path)
    cols = ["player_id", "player_display_name", "position", "team", "opponent_team", "season", "week", "season_type"]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = _numeric(df, PLAYER_STAT_COLUMNS)
    out = df[cols + list(PLAYER_STAT_COLUMNS)].copy()
    out = out.rename(columns={"player_display_name": "name", "opponent_team": "opponent"})
    out = out[out["player_id"].notna()]
    out["season_type"] = out["season_type"].fillna("REG")
    return out


def load_team_weeks(season: int, games: pd.DataFrame, *, refresh: bool = False) -> pd.DataFrame:
    path = download(PLAYER_STATS_TAG, TEAM_ASSET.format(season=season), refresh=refresh)
    df = pd.read_parquet(path)
    cols = ["team", "season", "week", "season_type", "opponent_team"]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = _numeric(df, TEAM_STAT_COLUMNS)
    out = df[cols + list(TEAM_STAT_COLUMNS)].copy()
    out = out.rename(columns={"opponent_team": "opponent"})
    out["season_type"] = out["season_type"].fillna("REG")

    # Points allowed drives the biggest single chunk of D/ST scoring, and it
    # only exists in the schedule, so join it on here.
    g = games[games["season"] == season]
    scored = pd.concat(
        [
            g[["season", "week", "home_team", "away_score"]].rename(
                columns={"home_team": "team", "away_score": "points_allowed"}
            ),
            g[["season", "week", "away_team", "home_score"]].rename(
                columns={"away_team": "team", "home_score": "points_allowed"}
            ),
        ],
        ignore_index=True,
    )
    out = out.merge(scored, on=["season", "week", "team"], how="left")
    out["points_allowed"] = pd.to_numeric(out["points_allowed"], errors="coerce")
    return out


# --- writes ---------------------------------------------------------------
def _insert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = list(df.columns)
    placeholders = ",".join("?" * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    rows = [tuple(None if pd.isna(v) else v for v in rec) for rec in df.itertuples(index=False, name=None)]
    conn.executemany(sql, rows)
    return len(rows)


def ingest(seasons: Iterable[int], db_path: Path | str | None = None, *, refresh: bool = False) -> dict[str, int]:
    """Download and store the given seasons. Safe to re-run."""
    seasons = list(seasons)
    counts = {"games": 0, "player_weeks": 0, "team_weeks": 0, "players": 0}
    games = load_games(refresh=refresh)
    with session(db_path) as conn:
        # games.parquet is small (every season since 1999) and the schedule for
        # the upcoming week is what tells us who is on bye, so store all of it.
        counts["games"] += _insert(conn, "games", games)
        for season in seasons:
            print(f"  {season}: players", end="", flush=True)
            try:
                pw = load_player_weeks(season, refresh=refresh)
            except FileNotFoundError as exc:
                print(f" skipped ({exc})")
                continue
            players = (
                pw.sort_values(["season", "week"])
                .groupby("player_id", as_index=False)
                .agg(name=("name", "last"), position=("position", "last"), last_team=("team", "last"), last_season=("season", "max"))
            )
            counts["players"] += _insert(conn, "players", players[["player_id", "name", "position", "last_team", "last_season"]])
            counts["player_weeks"] += _insert(
                conn,
                "player_weeks",
                pw[["player_id", "season", "week", "season_type", "team", "opponent", "position"] + list(PLAYER_STAT_COLUMNS)],
            )
            print(" teams", end="", flush=True)
            try:
                tw = load_team_weeks(season, games, refresh=refresh)
            except FileNotFoundError:
                tw = pd.DataFrame()
                print(" (no team file)", end="", flush=True)
            if not tw.empty:
                counts["team_weeks"] += _insert(
                    conn,
                    "team_weeks",
                    tw[["team", "season", "week", "season_type", "opponent", "points_allowed"] + list(TEAM_STAT_COLUMNS)],
                )
            conn.execute(
                "INSERT OR REPLACE INTO ingest_log (source, key, rows) VALUES (?,?,?)",
                ("nflverse", str(season), len(pw) + len(tw)),
            )
            print(" ok")
    return counts
