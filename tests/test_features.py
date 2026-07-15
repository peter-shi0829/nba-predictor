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
