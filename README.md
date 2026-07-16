# NBA Game Predictor

Predicts the winner of upcoming NBA games and shows the reasoning at
a public URL. A logistic regression trained on the last 10 seasons of
team stats: offensive and defensive rating, pace, recent form, and rest.

## How it works

1. A daily GitHub Action pulls team game logs with `nba_api`. The
   schedule endpoint (ScoreboardV2) is deprecated upstream; migrating
   to ScoreboardV3 is a tracked follow-up.
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
