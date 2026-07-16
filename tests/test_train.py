import numpy as np
import pandas as pd

from pipeline import train
from pipeline.features import MODEL_FEATURES


def synthetic_matchups(n=400, seed=3):
    """Matchups where NRTG_SEASON_DIFF mostly decides the winner."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(0.0, index=range(n), columns=MODEL_FEATURES)
    df["NRTG_SEASON_DIFF"] = rng.normal(0, 5, n)
    noise = rng.normal(0, 2, n)
    df["HOME_WIN"] = (df["NRTG_SEASON_DIFF"] + noise > 0).astype(int)
    df["SEASON"] = ["2022-23"] * (n // 2) + ["2023-24"] * (n - n // 2)
    return df


def test_split_by_season_is_time_ordered():
    m = synthetic_matchups()
    train_df, test_df = train.split_by_season(m, n_test_seasons=1)
    assert set(train_df["SEASON"]) == {"2022-23"}
    assert set(test_df["SEASON"]) == {"2023-24"}


def test_evaluate_beats_home_baseline_on_learnable_data():
    m = synthetic_matchups()
    model, metrics, test_df = train.evaluate(m, n_test_seasons=1)
    assert metrics["accuracy"] > metrics["baseline_home_accuracy"]
    assert 0 < metrics["log_loss"] < 1
    assert "PROB" in test_df.columns


def test_saved_model_round_trips(tmp_path):
    import joblib

    m = synthetic_matchups()
    model, _, _ = train.evaluate(m, n_test_seasons=1)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    probs = loaded.predict_proba(m[MODEL_FEATURES].head(5))
    assert probs.shape == (5, 2)
    assert "scaler" in loaded.named_steps
    assert "clf" in loaded.named_steps


def test_save_charts_writes_three_pngs(tmp_path):
    m = synthetic_matchups()
    model, metrics, test_df = train.evaluate(m, n_test_seasons=1)
    train.save_charts(model, test_df, chart_dir=tmp_path)
    names = {p.name for p in tmp_path.glob("*.png")}
    assert names == {"calibration.png", "coefficients.png", "accuracy_by_season.png"}
