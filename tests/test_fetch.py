from datetime import date

import pandas as pd

from pipeline import fetch


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
