"""Fetch and cache NBA team game logs, one parquet file per season."""
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIRST_SEASON_START = 2016
KEEP = ["SEASON", "IS_PLAYOFF", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID",
        "GAME_DATE", "MATCHUP", "WL", "PTS", "FGA", "FTA", "OREB", "TOV"]


def season_str(start_year):
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_season_start(today):
    """NBA seasons start in October."""
    return today.year if today.month >= 10 else today.year - 1


def seasons_for(today):
    """Ten completed seasons of history plus the in-progress season if any."""
    cur = current_season_start(today)
    first = max(FIRST_SEASON_START, cur - 10)
    return [season_str(y) for y in range(first, cur + 1)]


def fetch_season_raw(season, season_type):
    """One network call. Isolated so tests never touch it."""
    from nba_api.stats.endpoints import leaguegamelog
    for attempt in range(3):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star=season_type, timeout=60)
            df = log.get_data_frames()[0]
            time.sleep(1.5)  # stay polite, stats.nba.com rate-limits aggressively
            return df
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * 2 ** attempt)


def fetch_season(season):
    frames = []
    for season_type, flag in (("Regular Season", 0), ("Playoffs", 1)):
        df = fetch_season_raw(season, season_type)
        if df.empty:
            continue  # current season may have no playoff games yet
        df = df.copy()
        df["SEASON"] = season
        df["IS_PLAYOFF"] = flag
        frames.append(df[KEEP])
    return pd.concat(frames, ignore_index=True)


def season_path(data_dir, season):
    return Path(data_dir) / f"games_{season}.parquet"


def update_cache(data_dir=DATA_DIR, today=None):
    today = today or date.today()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    seasons = seasons_for(today)
    current = seasons[-1]
    for season in seasons:
        path = season_path(data_dir, season)
        if path.exists() and season != current:
            continue
        games = fetch_season(season)
        games.to_parquet(path, index=False)
        print(f"cached {season}: {len(games)} rows")


def load_games(data_dir=DATA_DIR):
    files = sorted(Path(data_dir).glob("games_*.parquet"))
    if not files:
        sys.exit("no cached data. run: python -m pipeline.fetch")
    games = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    return games


if __name__ == "__main__":
    update_cache()
