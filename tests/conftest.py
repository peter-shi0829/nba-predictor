import pandas as pd


def synthetic_games(n_games=6, season="2023-24", start="2024-01-02"):
    """Two teams, AAA and BBB, alternating home court every 2 days.

    Returns a raw team-game DataFrame with the same columns fetch.py caches.
    The home team always wins by a margin that grows with the game index,
    so tests have a deterministic home-court signal. Neither team has a
    persistent aggregate advantage.
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
