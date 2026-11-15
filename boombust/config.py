"""Runtime configuration, read from the environment with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional, only used to populate os.environ from a .env file
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "boombust.sqlite"
CACHE_DIR = REPO_ROOT / "data" / "cache"

# nflverse publishes one parquet per season as a GitHub release asset.
NFLVERSE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
NFLVERSE_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags"


@dataclass(frozen=True)
class Config:
    db_path: Path
    league_id: str | None
    season: int | None
    espn_s2: str | None
    swid: str | None

    @property
    def has_espn_credentials(self) -> bool:
        return bool(self.espn_s2 and self.swid)


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def load_config() -> Config:
    db = os.environ.get("BOOMBUST_DB")
    return Config(
        db_path=Path(db).expanduser() if db else DEFAULT_DB,
        league_id=os.environ.get("BOOMBUST_LEAGUE_ID") or None,
        season=_int_or_none(os.environ.get("BOOMBUST_SEASON")),
        espn_s2=os.environ.get("BOOMBUST_ESPN_S2") or None,
        swid=os.environ.get("BOOMBUST_SWID") or None,
    )
