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
