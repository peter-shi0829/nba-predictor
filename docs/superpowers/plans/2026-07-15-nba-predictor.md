# NBA Game Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public static website that shows win probabilities, stat comparisons, and plain-language reasoning for upcoming NBA games, powered by a logistic regression trained on 10 seasons of data and refreshed by a daily GitHub Action.

**Architecture:** A Python pipeline (fetch, features, train, predict) writes `site/predictions.json`. A hand-built static site (matchup spotlight layout, team color split style) renders that JSON. A daily GitHub Action reruns the pipeline and redeploys to GitHub Pages. No backend server.

**Tech Stack:** Python 3.12, nba_api, pandas, pyarrow, scikit-learn, seaborn/matplotlib, joblib, pytest. Vanilla HTML/CSS/JS for the site. GitHub Actions + GitHub Pages for automation and hosting.

**Spec:** `docs/superpowers/specs/2026-07-15-nba-predictor-design.md`

---

## File map

| File | Responsibility |
|---|---|
| `pipeline/fetch.py` | Pull team game logs from nba_api, cache one parquet per season, load the cache |
| `pipeline/features.py` | Compute ratings from box scores, leakage-safe team form features, matchup diff rows |
| `pipeline/train.py` | Time-split evaluation, seaborn charts, baseline sanity check, save final model |
| `pipeline/predict.py` | Team states as of today, upcoming schedule, factor explanations, write and validate `site/predictions.json` |
| `site/index.html`, `site/style.css`, `site/app.js` | Matchup spotlight UI, team color split cards |
| `site/teams.json` | Team names and colors for all 30 teams |
| `tests/` | Unit tests, leakage guard, payload schema test |
| `.github/workflows/daily.yml` | Daily fetch + predict, weekly retrain, deploy to Pages |
| `.github/workflows/ci.yml` | Run pytest on push |

Conventions used throughout:

- A "team-game" row is one team's box line for one game (two rows per game).
- All form features are computed strictly from games before the game in question.
- Feature columns: `STAT_COLS = ["ORTG", "DRTG", "NRTG", "PACE"]`, each with `_SEASON` and `_LAST10` variants, plus `REST_DAYS` and `BACK_TO_BACK`. Matchup rows hold home-minus-away `_DIFF` columns plus `IS_PLAYOFF`.

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`, `pipeline/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create the virtualenv and config files**

```bash
cd /Users/petershi/Projects/nba-predictor
python3 -m venv .venv
source .venv/bin/activate
mkdir -p pipeline tests site data models docs/charts .github/workflows
touch pipeline/__init__.py
```

`requirements.txt`:

```
nba_api
pandas
pyarrow
scikit-learn
seaborn
matplotlib
joblib
pytest
```

`pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

`.gitignore`:

```
__pycache__/
.pytest_cache/
.venv/
.DS_Store
*.egg-info/
.superpowers/
```

Note: `data/` and `models/` are committed on purpose. The daily Action needs the historical cache and model without refetching 10 seasons. Historical season files never change after the first fetch, so they cost one commit each.

`tests/conftest.py` (the synthetic data helper every test uses):

```python
import pandas as pd


def synthetic_games(n_games=6, season="2023-24", start="2024-01-02"):
    """Two teams, AAA and BBB, alternating home court every 2 days.

    Returns a raw team-game DataFrame with the same columns fetch.py caches.
    AAA generally outscores BBB so tests have a signal to find.
    """
    rows = []
    dates = pd.date_range(start, periods=n_games, freq="2D")
    for i, d in enumerate(dates):
        gid = f"G{i:04d}"
        home, away = ("AAA", "BBB") if i % 2 == 0 else ("BBB", "AAA")
        home_pts, away_pts = 100 + 2 * i, 96 + i
        base = {"SEASON": season, "IS_PLAYOFF": 0, "GAME_ID": gid, "GAME_DATE": d}
        rows.append({**base,
                     "TEAM_ID": 1 if home == "AAA" else 2,
                     "TEAM_ABBREVIATION": home,
                     "MATCHUP": f"{home} vs. {away}",
                     "WL": "W" if home_pts > away_pts else "L",
                     "PTS": home_pts, "FGA": 85 + i, "FTA": 20, "OREB": 10, "TOV": 12})
        rows.append({**base,
                     "TEAM_ID": 2 if home == "AAA" else 1,
                     "TEAM_ABBREVIATION": away,
                     "MATCHUP": f"{away} @ {home}",
                     "WL": "L" if home_pts > away_pts else "W",
                     "PTS": away_pts, "FGA": 88, "FTA": 18, "OREB": 9, "TOV": 14})
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Install dependencies and verify pytest runs**

Run: `pip install -r requirements.txt && pytest`
Expected: `no tests ran` (exit code 5 is fine at this point)

- [ ] **Step 3: Commit**

```bash
git add requirements.txt pytest.ini .gitignore pipeline/__init__.py tests/conftest.py
git commit -m "chore: project scaffolding"
```

---

### Task 2: Season helpers and the fetch cache

**Files:**
- Create: `pipeline/fetch.py`
- Test: `tests/test_fetch.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_fetch.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fetch.py -v`
Expected: FAIL with `ImportError` or `AttributeError: module 'pipeline.fetch' has no attribute 'seasons_for'`

- [ ] **Step 3: Implement `pipeline/fetch.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetch.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add pipeline/fetch.py tests/test_fetch.py
git commit -m "feat: season-aware game log fetching with parquet cache"
```

---

### Task 3: Ratings from box scores

**Files:**
- Create: `pipeline/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.features'`

- [ ] **Step 3: Implement `add_ratings` in `pipeline/features.py`**

```python
"""Turn raw team-game box lines into leakage-safe model features."""
import pandas as pd

STAT_COLS = ["ORTG", "DRTG", "NRTG", "PACE"]
FORM_COLS = ([f"{c}_{w}" for c in STAT_COLS for w in ("SEASON", "LAST10")]
             + ["REST_DAYS", "BACK_TO_BACK"])
DIFF_COLS = [f"{c}_DIFF" for c in FORM_COLS]
MODEL_FEATURES = DIFF_COLS + ["IS_PLAYOFF"]


def add_ratings(games):
    """Per team-game: possessions, offensive/defensive/net rating, pace."""
    g = games.copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    g["POSS"] = g["FGA"] - g["OREB"] + g["TOV"] + 0.44 * g["FTA"]
    opp = g[["GAME_ID", "TEAM_ID", "PTS", "POSS"]].rename(columns={
        "TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS", "POSS": "OPP_POSS"})
    merged = g.merge(opp, on="GAME_ID")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()
    merged["ORTG"] = 100 * merged["PTS"] / merged["POSS"]
    merged["DRTG"] = 100 * merged["OPP_PTS"] / merged["OPP_POSS"]
    merged["NRTG"] = merged["ORTG"] - merged["DRTG"]
    merged["PACE"] = (merged["POSS"] + merged["OPP_POSS"]) / 2
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: 1 PASSED

- [ ] **Step 5: Commit**

```bash
git add pipeline/features.py tests/test_features.py
git commit -m "feat: compute ratings and pace from box scores"
```

---

### Task 4: Team form features with the leakage guard

**Files:**
- Modify: `pipeline/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_features.py`)**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: 3 new FAIL with `AttributeError: module 'pipeline.features' has no attribute 'add_team_form'`

- [ ] **Step 3: Implement `add_team_form` (append to `pipeline/features.py`)**

```python
def add_team_form(rated):
    """Pre-game form for every team-game row.

    Every feature is shifted by one game so a row only ever sees games
    played strictly before it.
    """
    g = rated.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"]).copy()
    grp = g.groupby(["TEAM_ID", "SEASON"])
    for col in STAT_COLS:
        g[f"{col}_SEASON"] = grp[col].transform(
            lambda s: s.shift(1).expanding().mean())
        g[f"{col}_LAST10"] = grp[col].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    g["REST_DAYS"] = (grp["GAME_DATE"].diff().dt.days
                      .clip(upper=7).fillna(7.0))
    g["BACK_TO_BACK"] = (g["REST_DAYS"] <= 1).astype(int)
    return g
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add pipeline/features.py tests/test_features.py
git commit -m "feat: pre-game team form features with leakage guard test"
```

---

### Task 5: Matchup rows

**Files:**
- Modify: `pipeline/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_features.py`)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py::test_build_matchups_one_row_per_game_with_diffs -v`
Expected: FAIL with `AttributeError: module 'pipeline.features' has no attribute 'build_matchups'`

- [ ] **Step 3: Implement `build_matchups` (append to `pipeline/features.py`)**

```python
def build_matchups(formed):
    """One row per game: home-minus-away feature diffs plus the label."""
    home = formed[formed["MATCHUP"].str.contains("vs.", regex=False)]
    away = formed[formed["MATCHUP"].str.contains("@", regex=False)]
    m = home.merge(away, on="GAME_ID", suffixes=("_HOME", "_AWAY"))
    rows = pd.DataFrame({
        "GAME_ID": m["GAME_ID"],
        "GAME_DATE": m["GAME_DATE_HOME"],
        "SEASON": m["SEASON_HOME"],
        "IS_PLAYOFF": m["IS_PLAYOFF_HOME"],
        "HOME_TEAM": m["TEAM_ABBREVIATION_HOME"],
        "AWAY_TEAM": m["TEAM_ABBREVIATION_AWAY"],
        "HOME_PTS": m["PTS_HOME"],
        "AWAY_PTS": m["PTS_AWAY"],
        "HOME_WIN": (m["WL_HOME"] == "W").astype(int),
    })
    for col in FORM_COLS:
        rows[f"{col}_DIFF"] = m[f"{col}_HOME"] - m[f"{col}_AWAY"]
    return rows.dropna(subset=DIFF_COLS).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add pipeline/features.py tests/test_features.py
git commit -m "feat: matchup rows with home-away feature diffs"
```

---

### Task 6: Training, evaluation, and charts

**Files:**
- Create: `pipeline/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_train.py`:

```python
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


def test_save_charts_writes_three_pngs(tmp_path):
    m = synthetic_matchups()
    model, metrics, test_df = train.evaluate(m, n_test_seasons=1)
    train.save_charts(model, test_df, chart_dir=tmp_path)
    names = {p.name for p in tmp_path.glob("*.png")}
    assert names == {"calibration.png", "coefficients.png", "accuracy_by_season.png"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.train'`

- [ ] **Step 3: Implement `pipeline/train.py`**

```python
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
    save_charts(model, test_df)
    final = make_model().fit(matchups[MODEL_FEATURES], matchups["HOME_WIN"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add pipeline/train.py tests/test_train.py
git commit -m "feat: time-split training with evaluation charts and baseline gate"
```

---

### Task 7: Prediction core, team states and explanations

**Files:**
- Create: `pipeline/predict.py`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_predict.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.predict'`

- [ ] **Step 3: Implement the core of `pipeline/predict.py`**

```python
"""Score upcoming games and write site/predictions.json."""
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from pipeline.features import (DIFF_COLS, FORM_COLS, MODEL_FEATURES,
                               STAT_COLS, add_ratings, add_team_form,
                               build_matchups)
from pipeline.fetch import load_games
from pipeline.train import MODEL_PATH

SITE_PATH = Path(__file__).resolve().parent.parent / "site" / "predictions.json"
RETRO_GAMES = 8

LABELS = {
    "ORTG_SEASON_DIFF": "season-long offense",
    "DRTG_SEASON_DIFF": "season-long defense",
    "NRTG_SEASON_DIFF": "overall team strength",
    "PACE_SEASON_DIFF": "preferred pace",
    "ORTG_LAST10_DIFF": "recent offensive form",
    "DRTG_LAST10_DIFF": "recent defensive form",
    "NRTG_LAST10_DIFF": "recent form",
    "PACE_LAST10_DIFF": "recent pace",
    "REST_DAYS_DIFF": "the rest advantage",
    "BACK_TO_BACK_DIFF": "back-to-back fatigue",
    "IS_PLAYOFF": "playoff conditions",
}


def team_state(rated, abbr, season, game_date):
    """A team's pre-game aggregates as of game_date. None if no games yet."""
    tg = rated[(rated["TEAM_ABBREVIATION"] == abbr)
               & (rated["SEASON"] == season)
               & (rated["GAME_DATE"] < game_date)].sort_values("GAME_DATE")
    if tg.empty:
        return None
    state = {}
    for col in STAT_COLS:
        state[f"{col}_SEASON"] = tg[col].mean()
        state[f"{col}_LAST10"] = tg[col].tail(10).mean()
    rest = (pd.Timestamp(game_date) - tg["GAME_DATE"].max()).days
    state["REST_DAYS"] = float(min(rest, 7))
    state["BACK_TO_BACK"] = int(rest <= 1)
    return state


def matchup_features(home_state, away_state, is_playoff):
    row = {f"{k}_DIFF": home_state[k] - away_state[k] for k in FORM_COLS}
    row["IS_PLAYOFF"] = int(is_playoff)
    return row


def factor_contributions(model, row):
    x = pd.DataFrame([row])[MODEL_FEATURES]
    scaled = model.named_steps["scaler"].transform(x)[0]
    coefs = model.named_steps["clf"].coef_[0]
    return dict(zip(MODEL_FEATURES, scaled * coefs))


def explanation(home_abbr, away_abbr, prob_home, contributions):
    pick_home = prob_home >= 0.5
    pick = home_abbr if pick_home else away_abbr
    signed = {k: (v if pick_home else -v) for k, v in contributions.items()}
    top = [k for k, v in sorted(signed.items(), key=lambda kv: kv[1],
                                reverse=True) if v > 0][:2]
    labels = [LABELS[k] for k in top]
    if not labels:
        return f"A close one, but the model leans {pick}."
    if len(labels) == 1:
        return f"{pick} gets the edge here, mostly on {labels[0]}."
    return f"{pick} gets the edge here, built on {labels[0]} and {labels[1]}."


def side_dict(abbr, prob, state):
    stats = {"ortg": None, "drtg": None, "pace": None,
             "net_last10": None, "rest_days": None}
    if state is not None:
        stats = {
            "ortg": round(state["ORTG_SEASON"], 1),
            "drtg": round(state["DRTG_SEASON"], 1),
            "pace": round(state["PACE_SEASON"], 1),
            "net_last10": round(state["NRTG_LAST10"], 1),
            "rest_days": int(state["REST_DAYS"]),
        }
    return {"abbr": abbr, "win_prob": prob, "stats": stats}


def build_entry(model, home_abbr, away_abbr, home_state, away_state,
                is_playoff, date_str, time_et):
    if home_state is None or away_state is None:
        # opening-night fallback: no information, all diffs zero
        row = {c: 0.0 for c in DIFF_COLS}
        row["IS_PLAYOFF"] = int(is_playoff)
    else:
        row = matchup_features(home_state, away_state, is_playoff)
    prob_home = float(model.predict_proba(
        pd.DataFrame([row])[MODEL_FEATURES])[0, 1])
    ph = round(prob_home, 3)
    contrib = factor_contributions(model, row)
    return {
        "date": date_str,
        "time_et": time_et,
        "is_playoff": bool(is_playoff),
        "home": side_dict(home_abbr, ph, home_state),
        "away": side_dict(away_abbr, round(1 - ph, 3), away_state),
        "explanation": explanation(home_abbr, away_abbr, prob_home, contrib),
        "actual": None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_predict.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add pipeline/predict.py tests/test_predict.py
git commit -m "feat: prediction core with factor-based explanations"
```

---

### Task 8: Schedule fetch, payload validation, retro mode, and the predict entrypoint

**Files:**
- Modify: `pipeline/predict.py`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_predict.py`)**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_predict.py -v`
Expected: 3 new FAIL with `AttributeError: module 'pipeline.predict' has no attribute 'fetch_schedule'`

- [ ] **Step 3: Implement schedule, validation, retro, and main (append to `pipeline/predict.py`)**

```python
def _fetch_day_real(d):
    from nba_api.stats.endpoints import scoreboardv2
    for attempt in range(3):
        try:
            sb = scoreboardv2.ScoreboardV2(
                game_date=d.strftime("%m/%d/%Y"), timeout=60)
            time.sleep(1.5)
            return sb.game_header.get_data_frame()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * 2 ** attempt)


def fetch_schedule(days=7, start=None, fetch_day=_fetch_day_real):
    """Upcoming real games. GAME_ID prefix 002 = regular season, 004 = playoffs."""
    start = start or date.today()
    games = []
    for offset in range(days):
        d = start + timedelta(days=offset)
        header = fetch_day(d)
        for _, g in header.iterrows():
            gid = str(g["GAME_ID"])
            status = (g["GAME_STATUS_TEXT"] or "").strip()
            if gid[:3] not in ("002", "004") or status == "Final":
                continue
            games.append({
                "game_id": gid,
                "date": d.isoformat(),
                "time_et": status,
                "home_team_id": int(g["HOME_TEAM_ID"]),
                "away_team_id": int(g["VISITOR_TEAM_ID"]),
                "is_playoff": gid.startswith("004"),
            })
    return games


def _check(cond, msg):
    if not cond:
        raise ValueError(f"bad predictions payload: {msg}")


def validate_payload(payload):
    _check(payload["mode"] in ("upcoming", "retro"), "mode")
    datetime.fromisoformat(payload["generated_at"])
    stat_keys = {"ortg", "drtg", "pace", "net_last10", "rest_days"}
    for g in payload["games"]:
        for side in ("home", "away"):
            s = g[side]
            _check(0.0 <= s["win_prob"] <= 1.0, f"win_prob {s['win_prob']}")
            _check(set(s["stats"]) == stat_keys, "stats keys")
        _check(abs(g["home"]["win_prob"] + g["away"]["win_prob"] - 1) < 1e-6,
               "probs do not sum to 1")
        _check(g["home"]["abbr"] != g["away"]["abbr"], "same team twice")
        _check(isinstance(g["explanation"], str) and g["explanation"],
               "empty explanation")


def retro_entries(model, games, n=RETRO_GAMES):
    """Offseason mode: replay the last n playoff games, prediction vs result."""
    rated = add_ratings(games)
    matchups = build_matchups(add_team_form(rated))
    playoff = matchups[matchups["IS_PLAYOFF"] == 1].sort_values("GAME_DATE")
    entries = []
    for _, row in playoff.tail(n).iterrows():
        home_state = team_state(rated, row["HOME_TEAM"], row["SEASON"],
                                row["GAME_DATE"])
        away_state = team_state(rated, row["AWAY_TEAM"], row["SEASON"],
                                row["GAME_DATE"])
        entry = build_entry(model, row["HOME_TEAM"], row["AWAY_TEAM"],
                            home_state, away_state, is_playoff=True,
                            date_str=row["GAME_DATE"].date().isoformat(),
                            time_et="")
        winner = row["HOME_TEAM"] if row["HOME_WIN"] else row["AWAY_TEAM"]
        entry["actual"] = {"winner": winner,
                           "home_pts": int(row["HOME_PTS"]),
                           "away_pts": int(row["AWAY_PTS"])}
        entries.append(entry)
    return entries


def upcoming_entries(model, games, schedule):
    from nba_api.stats.static import teams as static_teams
    id_to_abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    rated = add_ratings(games)
    season = sorted(games["SEASON"].unique())[-1]
    entries = []
    for g in schedule:
        home_abbr = id_to_abbr[g["home_team_id"]]
        away_abbr = id_to_abbr[g["away_team_id"]]
        game_date = pd.Timestamp(g["date"])
        entries.append(build_entry(
            model, home_abbr, away_abbr,
            team_state(rated, home_abbr, season, game_date),
            team_state(rated, away_abbr, season, game_date),
            is_playoff=g["is_playoff"], date_str=g["date"],
            time_et=g["time_et"]))
    return entries


def main():
    games = load_games()
    model = joblib.load(MODEL_PATH)
    schedule = fetch_schedule()
    if schedule:
        mode, entries = "upcoming", upcoming_entries(model, games, schedule)
    else:
        mode, entries = "retro", retro_entries(model, games)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "games": entries,
    }
    validate_payload(payload)
    SITE_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {SITE_PATH}: mode={mode}, {len(entries)} games")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: all tests PASS (13 total)

- [ ] **Step 5: Commit**

```bash
git add pipeline/predict.py tests/test_predict.py
git commit -m "feat: schedule fetch, payload validation, retro mode, predict entrypoint"
```

---

### Task 9: Team colors and the site shell

**Files:**
- Create: `site/teams.json`, `site/index.html`, `site/style.css`

No unit tests for static assets; Task 10 verifies the rendered site in a browser.

- [ ] **Step 1: Create `site/teams.json`**

```json
{
  "ATL": {"name": "Hawks", "color": "#E03A3E"},
  "BOS": {"name": "Celtics", "color": "#007A33"},
  "BKN": {"name": "Nets", "color": "#111111"},
  "CHA": {"name": "Hornets", "color": "#1D1160"},
  "CHI": {"name": "Bulls", "color": "#CE1141"},
  "CLE": {"name": "Cavaliers", "color": "#860038"},
  "DAL": {"name": "Mavericks", "color": "#00538C"},
  "DEN": {"name": "Nuggets", "color": "#0E2240"},
  "DET": {"name": "Pistons", "color": "#C8102E"},
  "GSW": {"name": "Warriors", "color": "#1D428A"},
  "HOU": {"name": "Rockets", "color": "#CE1141"},
  "IND": {"name": "Pacers", "color": "#002D62"},
  "LAC": {"name": "Clippers", "color": "#C8102E"},
  "LAL": {"name": "Lakers", "color": "#552583"},
  "MEM": {"name": "Grizzlies", "color": "#5D76A9"},
  "MIA": {"name": "Heat", "color": "#98002E"},
  "MIL": {"name": "Bucks", "color": "#00471B"},
  "MIN": {"name": "Timberwolves", "color": "#0C2340"},
  "NOP": {"name": "Pelicans", "color": "#85714D"},
  "NYK": {"name": "Knicks", "color": "#F58426"},
  "OKC": {"name": "Thunder", "color": "#007AC1"},
  "ORL": {"name": "Magic", "color": "#0077C0"},
  "PHI": {"name": "76ers", "color": "#006BB6"},
  "PHX": {"name": "Suns", "color": "#E56020"},
  "POR": {"name": "Trail Blazers", "color": "#E03A3E"},
  "SAC": {"name": "Kings", "color": "#5A2D81"},
  "SAS": {"name": "Spurs", "color": "#000000"},
  "TOR": {"name": "Raptors", "color": "#CE1141"},
  "UTA": {"name": "Jazz", "color": "#002B5C"},
  "WAS": {"name": "Wizards", "color": "#E31837"}
}
```

- [ ] **Step 2: Create `site/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA Game Predictor</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header>
  <h1>NBA Predictor</h1>
  <p id="mode-note" class="hidden"></p>
</header>
<main>
  <button id="prev" class="arrow" aria-label="Previous game">&#8592;</button>
  <section id="card" aria-live="polite"></section>
  <button id="next" class="arrow" aria-label="Next game">&#8594;</button>
</main>
<div id="dots"></div>
<footer><span id="updated"></span></footer>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Create `site/style.css`**

```css
* { box-sizing: border-box; margin: 0; }

body {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: #0d0f14;
  color: #f2f4f8;
  font-family: "Helvetica Neue", Arial, sans-serif;
}

header { text-align: center; padding: 28px 16px 8px; }
header h1 { font-size: 22px; letter-spacing: 3px; text-transform: uppercase; }
#mode-note {
  margin-top: 10px; font-size: 13px; color: #c9cfda;
  background: #1a2030; display: inline-block; padding: 6px 14px;
  border-radius: 999px;
}
.hidden { display: none; }

main {
  flex: 1; display: flex; align-items: center; justify-content: center;
  gap: 14px; padding: 16px;
}

#card {
  position: relative;
  width: min(560px, 90vw);
  border-radius: 18px;
  padding: 26px 26px 22px;
  background: linear-gradient(105deg,
    var(--home-color, #1d428a) 49%, var(--away-color, #c8102e) 51%);
  box-shadow: 0 16px 40px rgba(0, 0, 0, .45);
}

.badge {
  position: absolute; top: -11px; left: 50%; transform: translateX(-50%);
  background: #f5b942; color: #1a1a1a; font-size: 11px; font-weight: 700;
  letter-spacing: 2px; padding: 4px 12px; border-radius: 999px;
}

.teams { display: flex; justify-content: space-between; align-items: baseline;
  text-shadow: 0 1px 3px rgba(0, 0, 0, .5); }
.team { font-size: 20px; font-weight: 700; }
.tip { font-size: 12px; opacity: .85; }

.probs { display: flex; justify-content: space-between; align-items: baseline;
  margin-top: 14px; text-shadow: 0 1px 3px rgba(0, 0, 0, .5); }
.probs b { font-size: 30px; }
.probs span { font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
  opacity: .75; }

.bar { background: rgba(255, 255, 255, .28); border-radius: 6px; height: 12px;
  margin: 12px 0 16px; overflow: hidden; }
.bar-fill { background: #fff; height: 100%; border-radius: 6px 0 0 6px; }

.stats { background: rgba(8, 10, 14, .55); border-radius: 12px;
  padding: 12px 16px; }
.stat-row { display: flex; justify-content: space-between; font-size: 13px;
  padding: 4px 0; }
.stat-label { color: #aeb6c4; font-size: 12px; }

.why { margin-top: 14px; font-size: 14px; line-height: 1.45;
  background: rgba(8, 10, 14, .55); border-radius: 12px; padding: 12px 16px; }

.actual { margin-top: 10px; font-size: 13px; text-align: center;
  background: rgba(8, 10, 14, .7); border-radius: 999px; padding: 8px 14px; }

.arrow {
  background: #1a2030; color: #f2f4f8; border: none; border-radius: 50%;
  width: 46px; height: 46px; font-size: 20px; cursor: pointer;
}
.arrow:hover { background: #273147; }
.arrow:disabled { opacity: .3; cursor: default; }

#dots { display: flex; justify-content: center; gap: 8px; padding: 14px; }
#dots span { width: 8px; height: 8px; border-radius: 50%; background: #333c4f; }
#dots span.on { background: #f2f4f8; }

footer { text-align: center; padding: 10px 0 22px; font-size: 12px;
  color: #7d8697; }

.empty { text-align: center; font-size: 15px; padding: 40px; color: #c9cfda; }

@media (max-width: 640px) {
  main { gap: 8px; }
  .arrow { width: 38px; height: 38px; }
  .probs b { font-size: 24px; }
}
```

- [ ] **Step 4: Commit**

```bash
git add site/teams.json site/index.html site/style.css
git commit -m "feat: site shell with team color split styling"
```

---

### Task 10: Site rendering logic

**Files:**
- Create: `site/app.js`, `site/predictions.json` (sample for local dev, overwritten by the pipeline)

- [ ] **Step 1: Create a sample `site/predictions.json`**

```json
{
  "generated_at": "2026-07-15T13:00:00+00:00",
  "mode": "retro",
  "games": [
    {
      "date": "2026-06-19", "time_et": "", "is_playoff": true,
      "home": {"abbr": "OKC", "win_prob": 0.58,
               "stats": {"ortg": 117.4, "drtg": 109.8, "pace": 99.2,
                         "net_last10": 5.1, "rest_days": 2}},
      "away": {"abbr": "NYK", "win_prob": 0.42,
               "stats": {"ortg": 115.1, "drtg": 111.6, "pace": 96.8,
                         "net_last10": 3.3, "rest_days": 2}},
      "explanation": "OKC gets the edge here, built on overall team strength and season-long defense.",
      "actual": {"winner": "OKC", "home_pts": 112, "away_pts": 104}
    },
    {
      "date": "2026-06-16", "time_et": "", "is_playoff": true,
      "home": {"abbr": "NYK", "win_prob": 0.55,
               "stats": {"ortg": 115.3, "drtg": 111.2, "pace": 96.9,
                         "net_last10": 4.0, "rest_days": 2}},
      "away": {"abbr": "OKC", "win_prob": 0.45,
               "stats": {"ortg": 117.1, "drtg": 110.1, "pace": 99.0,
                         "net_last10": 4.8, "rest_days": 2}},
      "explanation": "NYK gets the edge here, mostly on the rest advantage.",
      "actual": {"winner": "OKC", "home_pts": 101, "away_pts": 108}
    }
  ]
}
```

- [ ] **Step 2: Create `site/app.js`**

```javascript
let games = [];
let teams = {};
let idx = 0;

const pct = v => Math.round(v * 100) + "%";
const fmt = v => (v === null || v === undefined) ? "–" : v;

async function load() {
  const [pred, teamData] = await Promise.all([
    fetch("predictions.json").then(r => r.json()),
    fetch("teams.json").then(r => r.json()),
  ]);
  teams = teamData;
  games = pred.games;
  document.getElementById("updated").textContent =
    "Last updated " + new Date(pred.generated_at).toLocaleString();
  if (pred.mode === "retro") {
    const note = document.getElementById("mode-note");
    note.textContent =
      "Offseason. Looking back at the last playoffs: model pick vs what happened.";
    note.classList.remove("hidden");
  }
  if (!games.length) {
    document.getElementById("card").innerHTML =
      "<p class='empty'>No games in the next week. Check back soon.</p>";
    document.getElementById("prev").disabled = true;
    document.getElementById("next").disabled = true;
    return;
  }
  const dots = document.getElementById("dots");
  games.forEach(() => dots.appendChild(document.createElement("span")));
  render();
}

function teamName(abbr) {
  return (teams[abbr] && teams[abbr].name) || abbr;
}

function teamColor(abbr, fallback) {
  return (teams[abbr] && teams[abbr].color) || fallback;
}

function statRow(label, h, a) {
  return `<div class="stat-row"><span>${fmt(h)}</span>` +
         `<span class="stat-label">${label}</span><span>${fmt(a)}</span></div>`;
}

function pickWasRight(g) {
  const pickedHome = g.home.win_prob >= 0.5;
  return pickedHome === (g.actual.winner === g.home.abbr);
}

function render() {
  const g = games[idx];
  const card = document.getElementById("card");
  card.style.setProperty("--home-color", teamColor(g.home.abbr, "#1d428a"));
  card.style.setProperty("--away-color", teamColor(g.away.abbr, "#c8102e"));
  card.innerHTML = `
    ${g.is_playoff ? '<span class="badge">PLAYOFFS</span>' : ""}
    <div class="teams">
      <span class="team">${teamName(g.home.abbr)}</span>
      <span class="tip">${g.time_et || g.date}</span>
      <span class="team">${teamName(g.away.abbr)}</span>
    </div>
    <div class="probs">
      <b>${pct(g.home.win_prob)}</b>
      <span>win probability</span>
      <b>${pct(g.away.win_prob)}</b>
    </div>
    <div class="bar"><div class="bar-fill" style="width:${g.home.win_prob * 100}%"></div></div>
    <div class="stats">
      ${statRow("Offensive rating", g.home.stats.ortg, g.away.stats.ortg)}
      ${statRow("Defensive rating", g.home.stats.drtg, g.away.stats.drtg)}
      ${statRow("Pace", g.home.stats.pace, g.away.stats.pace)}
      ${statRow("Net rating, last 10", g.home.stats.net_last10, g.away.stats.net_last10)}
      ${statRow("Days of rest", g.home.stats.rest_days, g.away.stats.rest_days)}
    </div>
    <p class="why">${g.explanation}</p>
    ${g.actual ? `<p class="actual">Final: ${g.actual.winner} won ` +
      `${g.actual.home_pts}-${g.actual.away_pts}. ` +
      `${pickWasRight(g) ? "Model got it right." : "Model missed this one."}</p>` : ""}
  `;
  document.querySelectorAll("#dots span").forEach(
    (d, i) => d.classList.toggle("on", i === idx));
  document.getElementById("prev").disabled = idx === 0;
  document.getElementById("next").disabled = idx === games.length - 1;
}

function move(delta) {
  const next = idx + delta;
  if (next < 0 || next >= games.length) return;
  idx = next;
  render();
}

document.getElementById("prev").addEventListener("click", () => move(-1));
document.getElementById("next").addEventListener("click", () => move(1));
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") move(-1);
  if (e.key === "ArrowRight") move(1);
});

load();
```

Note: the retro score line reads home points then away points, matching the home-left card layout.

- [ ] **Step 3: Verify in a browser**

Run: `python3 -m http.server 8321 -d site`
Open http://localhost:8321 and check:
- Card shows OKC vs NYK with a diagonal color split (OKC blue left, NYK orange right)
- PLAYOFFS badge on top, retro note under the header
- Arrows and left/right keys switch between the two games, dots track position
- The "Final:" line shows on both games, one right and one missed
- Narrow the window below 640px, layout still fits

Stop the server with Ctrl+C when done.

- [ ] **Step 4: Commit**

```bash
git add site/app.js site/predictions.json
git commit -m "feat: matchup spotlight rendering with keyboard navigation"
```

---

### Task 11: README, CI, and the daily workflow

**Files:**
- Create: `README.md`, `.github/workflows/ci.yml`, `.github/workflows/daily.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI
on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - run: pytest -v
```

- [ ] **Step 2: Create `.github/workflows/daily.yml`**

```yaml
name: Daily predictions
on:
  schedule:
    - cron: "0 13 * * *"  # 9am ET during daylight time
  workflow_dispatch:

permissions:
  contents: write
  pages: write
  id-token: write

concurrency:
  group: daily
  cancel-in-progress: false

jobs:
  update:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Refresh data cache
        run: python -m pipeline.fetch
      - name: Retrain on Mondays or when the model is missing
        run: |
          if [ "$(date -u +%u)" = "1" ] || [ ! -f models/model.joblib ]; then
            python -m pipeline.train
          fi
      - name: Score upcoming games
        run: python -m pipeline.predict
      - name: Commit refreshed artifacts
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data models site/predictions.json docs/charts
          git diff --cached --quiet || git commit -m "chore: daily data refresh"
          git push
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
      - uses: actions/deploy-pages@v4
```

Known risk, decided in the spec's error handling section: stats.nba.com sometimes blocks datacenter IPs, which includes GitHub runners. If the daily job fails repeatedly on fetch timeouts, the fallback is running `python -m pipeline.fetch && python -m pipeline.predict && git push` from a local cron and letting the Action's push-triggered deploy handle publishing. Do not build this fallback now; wait to see if the block actually happens.

- [ ] **Step 3: Create `README.md`**

```markdown
# NBA Game Predictor

Predicts the winner of upcoming NBA games and shows the reasoning at
a public URL. A logistic regression trained on the last 10 seasons of
team stats: offensive and defensive rating, pace, recent form, and rest.

## How it works

1. A daily GitHub Action pulls team game logs with `nba_api`.
2. `pipeline/features.py` builds pre-game features for every matchup.
   Every number a prediction uses comes strictly from games played
   before tip-off.
3. A scikit-learn logistic regression outputs a home win probability.
   Its coefficient contributions become the plain-language explanation
   on each card.
4. The static site deploys to GitHub Pages. No server.

## Model honesty

Evaluated on the two most recent seasons, which the model never saw
during training. Vegas favorites win about 67% of games; a stats-only
model in the 63 to 66% range is performing as expected.

![Calibration](docs/charts/calibration.png)
![Coefficients](docs/charts/coefficients.png)
![Accuracy by season](docs/charts/accuracy_by_season.png)

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pipeline.fetch      # ~2 minutes first time, caches to data/
python -m pipeline.train      # prints held-out metrics, saves the model
python -m pipeline.predict    # writes site/predictions.json
python3 -m http.server 8321 -d site
```

## Tests

```bash
pytest
```
```

- [ ] **Step 4: Commit**

```bash
git add README.md .github/workflows/ci.yml .github/workflows/daily.yml
git commit -m "feat: CI, daily prediction workflow, README"
```

---

### Task 12: First real run and deployment

**Files:**
- Create: `data/games_*.parquet`, `models/model.joblib`, `docs/charts/*.png` (generated)
- Modify: `site/predictions.json` (generated)

- [ ] **Step 1: Run the full test suite one more time**

Run: `pytest -v`
Expected: all PASS

- [ ] **Step 2: Fetch 10 real seasons**

Run: `python -m pipeline.fetch`
Expected: ten `cached 20XX-YY: N rows` lines, roughly 2,400 to 2,700 rows per season. Takes a couple of minutes with the polite delays. If it fails with timeouts, wait a minute and rerun; already-cached seasons are skipped.

- [ ] **Step 3: Train on real data**

Run: `python -m pipeline.train`
Expected: printed metrics with `accuracy` between 0.62 and 0.68, above `baseline_home_accuracy` (usually around 0.54 to 0.57), `saved .../models/model.joblib`, and three PNGs in `docs/charts/`. If accuracy is below the baseline something is wrong in the features; stop and debug rather than shipping.

- [ ] **Step 4: Generate real predictions**

Run: `python -m pipeline.predict`
Expected (in July): `wrote .../site/predictions.json: mode=retro, 8 games` showing the 2026 Finals. During the season: `mode=upcoming` with that week's games.

- [ ] **Step 5: Eyeball the real site**

Run: `python3 -m http.server 8321 -d site`
Open http://localhost:8321. Check the retro banner, real team colors on the diagonal split, believable probabilities (nothing outside roughly 20 to 80%), and sensible explanation sentences. Stop the server.

- [ ] **Step 6: Commit the artifacts**

```bash
git add data models docs/charts site/predictions.json
git commit -m "feat: first trained model, cached seasons, real predictions"
```

- [ ] **Step 7: Create the GitHub repo and push**

```bash
gh repo create nba-predictor --public --source . --push
```

- [ ] **Step 8: Enable GitHub Pages via Actions and run the workflow**

```bash
gh api -X POST "repos/{owner}/nba-predictor/pages" -f build_type=actions
gh workflow run daily.yml
gh run watch
```

Expected: the run completes, and the site is live at `https://<username>.github.io/nba-predictor/`. If the fetch step fails with connection timeouts, that is the datacenter IP block described in Task 11; the site still deployed with the committed predictions, and the local-cron fallback is the fix.

- [ ] **Step 9: Verify the live site**

Open the Pages URL in a browser and repeat the Task 10 checks. Done means: live URL renders real predictions with the last-updated timestamp.

---

## Self-review notes

- Spec coverage: fetch/cache (Task 2), leakage-safe features (Tasks 3 to 5), model + evaluation + charts + baseline gate (Task 6), predictions with stats and explanations (Tasks 7 to 8), retro offseason mode (Task 8), matchup spotlight site in team color split (Tasks 9 to 10), daily Action + weekly retrain + Pages deploy + loud failure (Task 11), schema validation before ship (Task 8), all tests offline (conftest fixture).
- Deviation from spec: the spec's repo tree said `data/` is gitignored, but its data section said history is "cached in the repo". The plan commits per-season parquet files (historical ones never change) so the daily Action does not refetch 10 seasons; the spec text wins over the tree comment.
- Type consistency: `STAT_COLS`/`FORM_COLS`/`DIFF_COLS`/`MODEL_FEATURES` are defined once in `features.py` and imported everywhere; `team_state` returns keys matching `FORM_COLS`; `side_dict` stat keys match `validate_payload` and `app.js` usage.
