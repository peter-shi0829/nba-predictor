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
