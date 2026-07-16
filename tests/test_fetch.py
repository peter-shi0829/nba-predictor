from datetime import date

import pandas as pd
import pytest
from nba_api.stats.endpoints import leaguegamelog

from pipeline import fetch

RAW_COLS = [c for c in fetch.KEEP if c not in ("SEASON", "IS_PLAYOFF")]


def raw_frame(**overrides):
    """A minimal raw game-log frame with all the columns fetch_season keeps."""
    row = {"TEAM_ID": 1, "TEAM_ABBREVIATION": "AAA", "GAME_ID": "G1",
           "GAME_DATE": "2024-01-02", "MATCHUP": "AAA vs. BBB", "WL": "W",
           "PTS": 100, "FGA": 85, "FTA": 20, "OREB": 10, "TOV": 12}
    row.update(overrides)
    return pd.DataFrame([row])


def test_seasons_for_offseason_gives_ten_completed_seasons():
    seasons = fetch.seasons_for(date(2026, 7, 15))
    assert seasons[0] == "2016-17"
    assert seasons[-1] == "2025-26"
    assert len(seasons) == 10


def test_seasons_for_in_season_includes_current():
    seasons = fetch.seasons_for(date(2026, 11, 1))
    assert seasons[-1] == "2026-27"
    assert len(seasons) == 11


def test_update_cache_skips_cached_historical_seasons(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_season(season):
        calls.append(season)
        return pd.DataFrame({"SEASON": [season], "GAME_ID": ["G1"], "TEAM_ID": [1]})

    monkeypatch.setattr(fetch, "fetch_season", fake_fetch_season)
    today = date(2026, 7, 15)

    fetch.update_cache(data_dir=tmp_path, today=today)
    assert len(calls) == 10

    calls.clear()
    fetch.update_cache(data_dir=tmp_path, today=today)
    assert calls == ["2025-26"]  # only the current season refreshes


def test_fetch_season_empty_both_types_returns_empty_keep_frame(monkeypatch):
    monkeypatch.setattr(fetch, "fetch_season_raw",
                        lambda season, season_type: pd.DataFrame())
    df = fetch.fetch_season("2026-27")
    assert df.empty
    assert list(df.columns) == fetch.KEEP


def test_update_cache_writes_nothing_when_season_has_no_games(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "fetch_season_raw",
                        lambda season, season_type: pd.DataFrame())
    fetch.update_cache(data_dir=tmp_path, today=date(2026, 10, 5))
    seasons = fetch.seasons_for(date(2026, 10, 5))
    current_path = fetch.season_path(tmp_path, seasons[-1])
    assert not current_path.exists()
    assert list(tmp_path.glob("games_*.parquet")) == []


def test_fetch_season_tags_and_selects_columns(monkeypatch):
    def fake_raw(season, season_type):
        if season_type == "Regular Season":
            return raw_frame(JUNK="drop me")
        return pd.DataFrame()

    monkeypatch.setattr(fetch, "fetch_season_raw", fake_raw)
    df = fetch.fetch_season("2023-24")
    assert list(df.columns) == fetch.KEEP
    assert (df["SEASON"] == "2023-24").all()
    assert (df["IS_PLAYOFF"] == 0).all()
    assert "JUNK" not in df.columns


def test_load_games_combines_seasons_and_parses_dates(tmp_path):
    a = raw_frame(SEASON="2022-23", IS_PLAYOFF=0, GAME_DATE="2023-01-02")[fetch.KEEP]
    b = raw_frame(SEASON="2023-24", IS_PLAYOFF=0, GAME_DATE="2024-01-02")[fetch.KEEP]
    a.to_parquet(tmp_path / "games_2022-23.parquet", index=False)
    b.to_parquet(tmp_path / "games_2023-24.parquet", index=False)

    games = fetch.load_games(data_dir=tmp_path)
    assert len(games) == 2
    assert set(games["SEASON"]) == {"2022-23", "2023-24"}
    assert pd.api.types.is_datetime64_any_dtype(games["GAME_DATE"])


def test_load_games_exits_when_cache_empty(tmp_path):
    with pytest.raises(SystemExit):
        fetch.load_games(data_dir=tmp_path)


class FlakyGameLog:
    """Raises on the first `fail_times` constructions, then succeeds."""
    attempts = 0
    fail_times = 2
    frame = None

    def __init__(self, **kwargs):
        type(self).attempts += 1
        if type(self).attempts <= type(self).fail_times:
            raise ConnectionError("boom")

    def get_data_frames(self):
        return [type(self).frame]


def test_fetch_season_raw_retries_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(fetch.time, "sleep", sleeps.append)
    FlakyGameLog.attempts = 0
    FlakyGameLog.fail_times = 2
    FlakyGameLog.frame = raw_frame()
    monkeypatch.setattr(leaguegamelog, "LeagueGameLog", FlakyGameLog)

    df = fetch.fetch_season_raw("2023-24", "Regular Season")
    assert FlakyGameLog.attempts == 3
    assert df.equals(FlakyGameLog.frame)


def test_fetch_season_raw_raises_after_three_attempts(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    FlakyGameLog.attempts = 0
    FlakyGameLog.fail_times = 99
    monkeypatch.setattr(leaguegamelog, "LeagueGameLog", FlakyGameLog)

    with pytest.raises(ConnectionError):
        fetch.fetch_season_raw("2023-24", "Regular Season")
    assert FlakyGameLog.attempts == 3
