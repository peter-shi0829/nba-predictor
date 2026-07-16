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
    assert entry["explanation"] == ("No games played yet this season. "
                                    "This pick leans on home-court advantage alone.")


def test_explanation_names_the_picked_team():
    contributions = {"NRTG_SEASON_DIFF": 0.8, "REST_DAYS_DIFF": 0.3}
    text = predict.explanation("BOS", "NYK", 0.64, contributions)
    assert "BOS" in text
    text_away = predict.explanation("BOS", "NYK", 0.36, contributions)
    assert "NYK" in text_away


def test_fetch_schedule_filters_to_real_upcoming_games():
    def fake_day(d):
        return pd.DataFrame({
            "GAME_ID": ["0022600001", "0012600001", "0042500401"],
            "GAME_STATUS_TEXT": ["7:30 pm ET", "7:00 pm ET", "Final"],
            "HOME_TEAM_ID": [1610612738, 1610612752, 1610612743],
            "VISITOR_TEAM_ID": [1610612752, 1610612738, 1610612747],
        })

    games = predict.fetch_schedule(days=1, start=date(2026, 10, 20),
                                   fetch_day=fake_day)
    # preseason (001 prefix) and finished games are excluded
    assert len(games) == 1
    assert games[0]["game_id"] == "0022600001"
    assert games[0]["is_playoff"] is False


def test_fetch_schedule_handles_nan_status_text():
    def fake_day(d):
        return pd.DataFrame({
            "GAME_ID": ["0022600002"],
            "GAME_STATUS_TEXT": [float("nan")],
            "HOME_TEAM_ID": [1610612738],
            "VISITOR_TEAM_ID": [1610612752],
        })

    games = predict.fetch_schedule(days=1, start=date(2026, 10, 20),
                                   fetch_day=fake_day)
    assert len(games) == 1
    assert games[0]["game_id"] == "0022600002"
    assert games[0]["time_et"] == ""


def test_validate_payload_rejects_bad_probability():
    good = {
        "generated_at": "2026-07-15T13:00:00+00:00",
        "mode": "upcoming",
        "games": [{
            "date": "2026-10-20", "time_et": "7:30 pm ET", "is_playoff": False,
            "home": {"abbr": "BOS", "win_prob": 0.64,
                     "stats": {"ortg": 118.2, "drtg": 110.1, "pace": 98.0,
                               "net_last10": 6.4, "rest_days": 2}},
            "away": {"abbr": "NYK", "win_prob": 0.36,
                     "stats": {"ortg": 114.9, "drtg": 112.3, "pace": 97.1,
                               "net_last10": 1.2, "rest_days": 1}},
            "explanation": "BOS gets the edge here.",
            "actual": None,
        }],
    }
    predict.validate_payload(good)  # must not raise

    bad = {**good, "games": [{**good["games"][0],
                              "home": {**good["games"][0]["home"],
                                       "win_prob": 1.4}}]}
    try:
        predict.validate_payload(bad)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_retro_entries_include_actual_results():
    games = synthetic_games(n_games=8)
    games.loc[games.index[-4:], "IS_PLAYOFF"] = 1  # last two games are playoffs
    entries = predict.retro_entries(trained_model(), games, n=2)
    assert len(entries) == 2
    for e in entries:
        assert e["actual"]["winner"] in (e["home"]["abbr"], e["away"]["abbr"])
        assert e["is_playoff"] is True


def test_upcoming_entries_builds_valid_payload(monkeypatch):
    from nba_api.stats.static import teams as static_teams

    monkeypatch.setattr(static_teams, "get_teams",
                        lambda: [{"id": 1, "abbreviation": "AAA"},
                                 {"id": 2, "abbreviation": "BBB"}])
    games = synthetic_games(n_games=6)
    schedule = [{"game_id": "0022300100", "date": "2024-01-20",
                 "time_et": "7:30 pm ET", "home_team_id": 1,
                 "away_team_id": 2, "is_playoff": False}]
    entries = predict.upcoming_entries(trained_model(), games, schedule)
    assert len(entries) == 1
    assert set(entries[0]) == {"date", "time_et", "is_playoff", "home",
                               "away", "explanation", "actual"}
    assert entries[0]["actual"] is None
    predict.validate_payload({
        "generated_at": "2026-07-15T13:00:00+00:00",
        "mode": "upcoming",
        "games": entries,
    })
