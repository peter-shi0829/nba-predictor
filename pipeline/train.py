"""Train and evaluate the logistic regression, save charts and the model."""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.features import (MODEL_FEATURES, add_ratings, add_team_form,
                               build_matchups)
from pipeline.fetch import load_games

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
CHART_DIR = Path(__file__).resolve().parent.parent / "docs" / "charts"
N_TEST_SEASONS = 2


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def split_by_season(matchups, n_test_seasons=N_TEST_SEASONS):
    seasons = sorted(matchups["SEASON"].unique())
    test_seasons = set(seasons[-n_test_seasons:])
    is_test = matchups["SEASON"].isin(test_seasons)
    return matchups[~is_test], matchups[is_test]


def evaluate(matchups, n_test_seasons=N_TEST_SEASONS):
    train_df, test_df = split_by_season(matchups, n_test_seasons)
    model = make_model()
    model.fit(train_df[MODEL_FEATURES], train_df["HOME_WIN"])
    probs = model.predict_proba(test_df[MODEL_FEATURES])[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(test_df["HOME_WIN"], preds),
        "log_loss": log_loss(test_df["HOME_WIN"], probs),
        # picking the home team every time is the bar to clear
        "baseline_home_accuracy": test_df["HOME_WIN"].mean(),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    return model, metrics, test_df.assign(PROB=probs)


def save_charts(model, test_df, chart_dir=CHART_DIR):
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    frac_pos, mean_pred = calibration_curve(
        test_df["HOME_WIN"], test_df["PROB"], n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    sns.lineplot(x=mean_pred, y=frac_pos, marker="o", ax=ax)
    ax.set(xlabel="Predicted home win probability",
           ylabel="Actual home win rate",
           title="Calibration on held-out seasons")
    fig.savefig(chart_dir / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    coefs = pd.Series(model.named_steps["clf"].coef_[0],
                      index=MODEL_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=coefs.values, y=coefs.index, ax=ax, color="#1d428a")
    ax.set(xlabel="Coefficient (standardized features)",
           title="What the model weighs")
    fig.savefig(chart_dir / "coefficients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    correct = ((test_df["PROB"] >= 0.5).astype(int) == test_df["HOME_WIN"])
    by_season = (correct.groupby(test_df["SEASON"]).mean()
                 .reset_index(name="accuracy"))
    by_season.columns = ["SEASON", "accuracy"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=by_season, x="SEASON", y="accuracy", ax=ax, color="#1d428a")
    ax.axhline(test_df["HOME_WIN"].mean(), linestyle="--", color="gray",
               label="always pick home")
    ax.legend()
    ax.set(title="Accuracy by held-out season")
    fig.savefig(chart_dir / "accuracy_by_season.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main():
    games = load_games()
    matchups = build_matchups(add_team_form(add_ratings(games)))
    model, metrics, test_df = evaluate(matchups)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    if metrics["accuracy"] <= metrics["baseline_home_accuracy"]:
        sys.exit("model does not beat the home-court baseline; refusing to save")
    final = make_model().fit(matchups[MODEL_FEATURES], matchups["HOME_WIN"])
    save_charts(final, test_df)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
