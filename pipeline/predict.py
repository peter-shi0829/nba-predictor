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
    missing_state = home_state is None or away_state is None
    if missing_state:
        # opening-night fallback: no information, all diffs zero
        row = {c: 0.0 for c in DIFF_COLS}
        row["IS_PLAYOFF"] = int(is_playoff)
    else:
        row = matchup_features(home_state, away_state, is_playoff)
    prob_home = float(model.predict_proba(
        pd.DataFrame([row])[MODEL_FEATURES])[0, 1])
    ph = round(prob_home, 3)
    if missing_state:
        # a zero-diff row has no factors worth naming; be honest about it
        text = ("No games played yet this season. "
                "This pick leans on home-court advantage alone.")
    else:
        contrib = factor_contributions(model, row)
        text = explanation(home_abbr, away_abbr, prob_home, contrib)
    return {
        "date": date_str,
        "time_et": time_et,
        "is_playoff": bool(is_playoff),
        "home": side_dict(home_abbr, ph, home_state),
        "away": side_dict(away_abbr, round(1 - ph, 3), away_state),
        "explanation": text,
        "actual": None,
    }


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
            status = "" if pd.isna(g["GAME_STATUS_TEXT"]) else str(g["GAME_STATUS_TEXT"]).strip()
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
