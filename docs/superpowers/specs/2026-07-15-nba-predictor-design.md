# NBA Game Predictor: Design

Date: 2026-07-15
Status: Approved pending user review

## Purpose

A public website that predicts the winner of upcoming NBA games. Primary use is live prediction during the season and playoffs. Secondary use is as a portfolio piece, so the site must look clean and intentionally designed, not like a data science demo.

## Requirements

- Show upcoming NBA games automatically with a win probability for each. No user input required.
- Cover the regular season and playoffs, with playoff games visually highlighted. The site stays useful in October through June.
- Each game shows three things: the win probability, a side-by-side comparison of the key stats behind the pick, and a short explanation of which factors drove the prediction.
- Deployed to a free public URL.
- Tech stack: nba_api, pandas, scikit-learn, seaborn for the modeling pipeline. Hand-written HTML/CSS/vanilla JS for the site.

## Architecture

Static site plus a scheduled pipeline. There is no backend server.

1. A GitHub Action runs every morning (about 9am ET).
2. The Python pipeline pulls fresh data via nba_api, updates features, scores today's games, and writes `predictions.json`.
3. The static site is redeployed to GitHub Pages, reading `predictions.json` at load time.

Rationale: predictions only change once per day, after the previous night's games settle. A daily batch job gives the same freshness as a server with zero hosting cost and nothing to maintain.

## Data and model

**Data.** Team game logs for the last 10 seasons (2016-17 through 2025-26), regular season and playoffs, pulled with nba_api. Roughly 25,000 team-game rows. Historical seasons are fetched once and cached in the repo as parquet files; the daily job only fetches games since the last cached date.

**Features.** One row per game, built strictly from information available before tip-off:

- Season-to-date offensive rating, defensive rating, net rating, and pace for each team
- The same four stats over each team's last 10 games (recent form)
- Rest days since each team's previous game, and a back-to-back flag
- Home court indicator
- Playoff game indicator

Each matchup row is expressed as home-minus-away differences.

**Model.** A scikit-learn pipeline: StandardScaler then LogisticRegression. Logistic regression outputs a calibrated-ish win probability directly, and its coefficients times the feature values give per-game factor contributions, which power the "why this pick" text.

**Evaluation.** Time-based split: train on the first 8 seasons, test on the most recent 2. No shuffling, so evaluation honestly simulates predicting future games. Metrics: accuracy, log loss, and a calibration curve. Seaborn charts (calibration plot, coefficient importance, accuracy by season) are saved to `docs/charts/` and embedded in the README. Expectation setting: Vegas hits about 67% on winners; a stats-only logistic regression in the 63-66% range is a good result.

**Retraining.** The model retrains weekly inside the Action. Daily runs only re-score upcoming games with the existing model.

## Website

**Layout: matchup spotlight.** One game fills the viewport at a time with everything visible: teams, win probability bar, stat comparison rows, and the explanation sentence. Arrow keys, on-screen arrows, or scroll move between games, ordered by tip-off time. A small index dot row shows position (game 2 of 5). Playoff games get a "PLAYOFFS" badge.

**Visual style: team color split.** Each matchup card is split diagonally in the two teams' primary colors, with stats in a translucent dark panel for readability. Loud, fun, unmistakably NBA. Team colors come from a static JSON map of all 30 teams.

**Content per game:**

- Team names, tip-off time (ET), and win probability for each side
- A probability bar splitting the card width
- Stat rows: offensive rating, defensive rating, pace, recent form, rest days
- One or two sentences of explanation generated from the top absolute factor contributions, written in plain language
- "Last updated" timestamp in the footer, so stale data is never silently presented as fresh

**Offseason state.** When no games are scheduled within the next week, the site shows the most recent playoff games with the model's prediction next to the actual result, labeled clearly as a retrospective. This doubles as accuracy proof for portfolio viewers.

## Repo structure

```
nba-predictor/
  pipeline/
    fetch.py        # pull and cache game logs
    features.py     # build leakage-safe matchup rows
    train.py        # fit, evaluate, save model + charts
    predict.py      # score upcoming games -> site/predictions.json
  site/
    index.html
    style.css
    app.js
    teams.json      # team colors and metadata
    predictions.json  # generated artifact
  data/             # cached parquet, gitignored except a small fixture
  models/           # saved sklearn pipeline (joblib)
  tests/
  docs/charts/      # seaborn evaluation charts for the README
  docs/superpowers/specs/
  .github/workflows/daily.yml
```

## Error handling

- nba_api is unofficial and rate-limited. All fetches retry with exponential backoff and a polite delay between requests.
- If the daily fetch still fails, the Action fails loudly (GitHub sends an email) and no deploy happens. The site keeps serving yesterday's predictions with the visible "last updated" timestamp.
- If a team has fewer than 10 games played (early season), recent-form features fall back to season-to-date values.
- `predict.py` validates `predictions.json` against a schema before deploy so a malformed artifact never ships.

## Testing

- Unit tests for `features.py`, including a leakage guard test: features for a given game must be identical whether or not later games exist in the dataset.
- Fixture-based test of the full predict path using a small cached dataset, so CI never depends on the live NBA API.
- A schema test for `predictions.json`.
- Model sanity test: trained model must beat a home-court-always baseline on the held-out seasons.

## Out of scope (for now)

- Player-level features, injuries, betting odds
- A matchup picker for hypothetical games
- Series-level playoff predictions (who wins the series)
- Accounts, favorites, notifications
