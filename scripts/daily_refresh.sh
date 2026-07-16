#!/bin/zsh
# Daily data refresh, run by launchd on Peter's Mac because stats.nba.com
# blocks GitHub Actions runner IPs. Pushing the refreshed site/ triggers
# the Pages deploy workflow.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

git pull --rebase --quiet
.venv/bin/python -m pipeline.fetch
if [ "$(date -u +%u)" = "1" ] || [ ! -f models/model.joblib ]; then
  .venv/bin/python -m pipeline.train
fi
.venv/bin/python -m pipeline.predict
git add data models site/predictions.json docs/charts
git diff --cached --quiet || {
  git commit -m "chore: daily data refresh"
  git push
}
