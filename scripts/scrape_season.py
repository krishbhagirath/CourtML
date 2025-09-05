# scoreboard_full.py
# Usage:
#   python scoreboard_full.py --date 01/15/2025
#   python scoreboard_full.py              # defaults to today
#
# deps: nba_api pandas python-dateutil

import argparse
from datetime import datetime
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2


def fetch_scoreboard(date_str: str) -> pd.DataFrame:
    sb = scoreboardv2.ScoreboardV2(game_date=date_str)

    game_header, linescore, *_ = sb.get_data_frames()
    game_header.columns = [c.upper() for c in game_header.columns]
    linescore.columns = [c.upper() for c in linescore.columns]

    # Merge home
    home = linescore.rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "TEAM_ABBREVIATION": "HOME_ABBREV",
            "PTS": "HOME_PTS",
        }
    )
    gh_home = pd.merge(
        game_header,
        home[["GAME_ID", "HOME_TEAM_ID", "HOME_ABBREV", "HOME_PTS"]],
        on=["GAME_ID", "HOME_TEAM_ID"],
        how="left",
    )

    # Merge away
    away = linescore.rename(
        columns={
            "TEAM_ID": "VISITOR_TEAM_ID",
            "TEAM_ABBREVIATION": "AWAY_ABBREV",
            "PTS": "AWAY_PTS",
        }
    )
    out = pd.merge(
        gh_home,
        away[["GAME_ID", "VISITOR_TEAM_ID", "AWAY_ABBREV", "AWAY_PTS"]],
        on=["GAME_ID", "VISITOR_TEAM_ID"],
        how="left",
    )

    keep = [
        "GAME_ID",
        "GAME_DATE_EST",
        "GAME_STATUS_TEXT",
        "GAME_TIME",  # scheduled local tipoff time (ET)
        "AWAY_ABBREV",
        "AWAY_PTS",
        "HOME_ABBREV",
        "HOME_PTS",
        "ARENA_NAME",
    ]
    existing = [c for c in keep if c in out.columns]
    out = out[existing].copy()

    # Clean types
    if "GAME_DATE_EST" in out.columns:
        out["GAME_DATE_EST"] = pd.to_datetime(out["GAME_DATE_EST"], errors="coerce")
    for c in ["AWAY_PTS", "HOME_PTS"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NBA scoreboard for a given date.")
    ap.add_argument("--date", help='Date in MM/DD/YYYY (default: today)')
    args = ap.parse_args()

    date_str = args.date or datetime.today().strftime("%m/%d/%Y")
    df = fetch_scoreboard(date_str)

    if df.empty:
        print(f"No games for {date_str}")
    else:
        print(f"\nNBA Games on {date_str}\n")
        print(
            df.to_string(
                index=False,
                columns=[
                    "GAME_DATE_EST",
                    "GAME_TIME",
                    "AWAY_ABBREV",
                    "AWAY_PTS",
                    "HOME_ABBREV",
                    "HOME_PTS",
                    "GAME_STATUS_TEXT",
                    "ARENA_NAME",
                ],
            )
        )
