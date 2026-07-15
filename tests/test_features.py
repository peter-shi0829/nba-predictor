import pandas as pd
import pytest

from pipeline import features
from tests.conftest import synthetic_games


def test_add_ratings_computes_ortg_drtg_pace():
    games = synthetic_games(n_games=1)
    rated = features.add_ratings(games)

    assert len(rated) == 2  # still one row per team per game
    home = rated[rated["TEAM_ABBREVIATION"] == "AAA"].iloc[0]
    # possessions = FGA - OREB + TOV + 0.44 * FTA = 85 - 10 + 12 + 8.8 = 95.8
    assert home["POSS"] == pytest.approx(95.8)
    assert home["ORTG"] == pytest.approx(100 * 100 / 95.8)
    # opponent: 88 - 9 + 14 + 0.44 * 18 = 100.92 possessions, 96 points
    assert home["DRTG"] == pytest.approx(100 * 96 / 100.92)
    assert home["NRTG"] == pytest.approx(home["ORTG"] - home["DRTG"])
    assert home["PACE"] == pytest.approx((95.8 + 100.92) / 2)


def test_form_features_use_only_prior_games():
    games = synthetic_games(n_games=3)
    formed = features.add_team_form(features.add_ratings(games))
    aaa = formed[formed["TEAM_ABBREVIATION"] == "AAA"].sort_values("GAME_DATE")

    # First game of the season has no history
    assert pd.isna(aaa.iloc[0]["ORTG_SEASON"])
    # Second game sees exactly the first game's rating
    assert aaa.iloc[1]["ORTG_SEASON"] == pytest.approx(aaa.iloc[0]["ORTG"])
    # Third game sees the mean of the first two
    expected = (aaa.iloc[0]["ORTG"] + aaa.iloc[1]["ORTG"]) / 2
    assert aaa.iloc[2]["ORTG_SEASON"] == pytest.approx(expected)


def test_rest_days_and_back_to_back():
    games = synthetic_games(n_games=3)  # games every 2 days
    formed = features.add_team_form(features.add_ratings(games))
    aaa = formed[formed["TEAM_ABBREVIATION"] == "AAA"].sort_values("GAME_DATE")
    assert aaa.iloc[0]["REST_DAYS"] == 7  # unknown rest capped at 7
    assert aaa.iloc[1]["REST_DAYS"] == 2
    assert aaa.iloc[1]["BACK_TO_BACK"] == 0


def test_leakage_guard_future_games_do_not_change_features():
    """The classic bug this project must never have: features for a game
    must be identical whether or not later games exist in the dataset."""
    games = synthetic_games(n_games=6)
    cutoff = pd.Timestamp("2024-01-08")  # game 4 of 6

    full = features.add_team_form(features.add_ratings(games))
    truncated = features.add_team_form(
        features.add_ratings(games[games["GAME_DATE"] <= cutoff]))

    checked = 0
    for _, t_row in truncated[truncated["GAME_DATE"] == cutoff].iterrows():
        f_row = full[(full["GAME_DATE"] == cutoff)
                     & (full["TEAM_ID"] == t_row["TEAM_ID"])].iloc[0]
        for col in features.FORM_COLS:
            assert f_row[col] == pytest.approx(t_row[col], nan_ok=True), col
            checked += 1
    assert checked > 0


def test_build_matchups_one_row_per_game_with_diffs():
    games = synthetic_games(n_games=4)
    matchups = features.build_matchups(
        features.add_team_form(features.add_ratings(games)))

    # Game 1 drops: both teams have NaN season-to-date features
    assert len(matchups) == 3
    assert set(features.DIFF_COLS).issubset(matchups.columns)

    row = matchups.sort_values("GAME_DATE").iloc[0]  # game 2: BBB hosts AAA
    assert row["HOME_TEAM"] == "BBB"
    assert row["AWAY_TEAM"] == "AAA"
    assert row["HOME_WIN"] == 1  # home team scored 102 vs 97
    assert "HOME_PTS" in matchups.columns  # kept for retro mode display
