from datetime import date

import pandas as pd

from pipeline import predict, train
from pipeline.features import add_ratings
from tests.conftest import synthetic_games
from tests.test_train import synthetic_matchups


def trained_model():
    model, _, _ = train.evaluate(synthetic_matchups(), n_test_seasons=1)
    return model


def test_team_state_aggregates_prior_games_only():
    rated = add_ratings(synthetic_games(n_games=4))
    state = predict.team_state(rated, "AAA", "2023-24", pd.Timestamp("2024-01-06"))
    prior = rated[(rated["TEAM_ABBREVIATION"] == "AAA")
                  & (rated["GAME_DATE"] < "2024-01-06")]
    assert state["ORTG_SEASON"] == prior["ORTG"].mean()
    assert state["REST_DAYS"] == 2
    assert state["BACK_TO_BACK"] == 0


def test_team_state_none_when_no_games():
    rated = add_ratings(synthetic_games(n_games=2))
    assert predict.team_state(rated, "AAA", "2023-24",
                              pd.Timestamp("2024-01-01")) is None


def test_build_entry_probabilities_sum_to_one():
    rated = add_ratings(synthetic_games(n_games=6))
    home = predict.team_state(rated, "AAA", "2023-24", pd.Timestamp("2024-01-20"))
    away = predict.team_state(rated, "BBB", "2023-24", pd.Timestamp("2024-01-20"))
    entry = predict.build_entry(trained_model(), "AAA", "BBB", home, away,
                                is_playoff=False, date_str="2024-01-20",
                                time_et="7:30 pm ET")
    assert abs(entry["home"]["win_prob"] + entry["away"]["win_prob"] - 1) < 1e-9
    assert entry["explanation"]
    assert entry["home"]["stats"]["rest_days"] >= 0


def test_build_entry_with_missing_state_falls_back_to_even():
    entry = predict.build_entry(trained_model(), "AAA", "BBB", None, None,
                                is_playoff=False, date_str="2024-01-01",
                                time_et="")
    # all-zero diffs: probability should be near the home-court prior, not extreme
    assert 0.3 < entry["home"]["win_prob"] < 0.8
    assert entry["home"]["stats"]["ortg"] is None


def test_explanation_names_the_picked_team():
    contributions = {"NRTG_SEASON_DIFF": 0.8, "REST_DAYS_DIFF": 0.3}
    text = predict.explanation("BOS", "NYK", 0.64, contributions)
    assert "BOS" in text
    text_away = predict.explanation("BOS", "NYK", 0.36, contributions)
    assert "NYK" in text_away
