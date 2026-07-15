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
