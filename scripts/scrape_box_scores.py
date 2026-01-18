# scripts/extract_features_by_date.py
# Usage:
#   python scripts/extract_features_by_date.py --date 01/15/2025 --out features_2025_01_15.csv
#
# Deps:
#   pip install nba_api pandas

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd

from nba_api.stats.endpoints import (
    scoreboardv2,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
    boxscorefourfactorsv2,
)

RATE_DELAY = 0.35  # short pause between API calls (be polite)


def _frames_by_name(endpoint_obj) -> Dict[str, pd.DataFrame]:
    """Return {resultSetName: DataFrame} safely for any stats endpoint."""
    d = endpoint_obj.get_dict()
    out = {}
    for rs in d.get("resultSets", []):
        name = rs.get("name", "Unknown")
        headers = rs.get("headers", [])
        rows = rs.get("rowSet", [])
        out[name] = pd.DataFrame(rows, columns=headers)
    return out


def _scoreboard_game_headers(date_str: str) -> pd.DataFrame:
    sb = scoreboardv2.ScoreboardV2(game_date=date_str)
    frames = sb.get_data_frames()
    if not frames:
        return pd.DataFrame()
    gh = frames[0].copy()  # GameHeader
    gh.columns = [c.upper() for c in gh.columns]
    return gh


def _final_game_ids(date_str: str) -> List[str]:
    gh = _scoreboard_game_headers(date_str)
    if gh.empty:
        return []
    finals = gh[gh["GAME_STATUS_TEXT"].astype(str).str.strip().str.lower() == "final"]
    return [str(gid) for gid in finals["GAME_ID"].tolist()]


def _parse_min_to_float(min_str: Optional[str]) -> Optional[float]:
    """Convert 'MM:SS' (or 'M:SS') to minutes float. Returns None if not parseable."""
    if not isinstance(min_str, str) or ":" not in min_str:
        return None
    try:
        mm, ss = min_str.split(":")
        return int(mm) + int(ss) / 60.0
    except Exception:
        return None


def _minutes_weighted_mean(df: pd.DataFrame, value_col: str, min_col: str = "MIN") -> Optional[float]:
    """Weighted mean of value_col by minutes; returns None if not possible."""
    if df.empty or value_col not in df.columns or min_col not in df.columns:
        return None
    mins = df[min_col].apply(_parse_min_to_float)
    vals = pd.to_numeric(df[value_col], errors="coerce")
    mask = mins.notna() & vals.notna()
    if not mask.any():
        return None
    w = mins[mask]
    v = vals[mask]
    denom = w.sum()
    return float((v * w).sum() / denom) if denom > 0 else None


def _safe_max(df: pd.DataFrame, col: str) -> Optional[float]:
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().any():
        return float(s.max())
    return None


def _load_boxscore_sets(game_id: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fetch Traditional, Advanced, FourFactors; return dict of name -> frames_by_name."""
    out = {}
    try:
        tr = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        out["traditional"] = _frames_by_name(tr)
    except Exception as e:
        print(f"[warn] Traditional failed for {game_id}: {e}")
        out["traditional"] = {}
    time.sleep(RATE_DELAY)

    try:
        adv = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        out["advanced"] = _frames_by_name(adv)
    except Exception as e:
        print(f"[warn] Advanced failed for {game_id}: {e}")
        out["advanced"] = {}
    time.sleep(RATE_DELAY)

    try:
        ff = boxscorefourfactorsv2.BoxScoreFourFactorsV2(game_id=game_id)
        out["fourfactors"] = _frames_by_name(ff)
    except Exception as e:
        print(f"[warn] FourFactors failed for {game_id}: {e}")
        out["fourfactors"] = {}

    return out


def _team_rows_for_game(game_id: str, date_str: str) -> List[Dict]:
    """
    Build two rows (one per team) containing requested features + opponent features.
    """
    data = _load_boxscore_sets(game_id)

    # --- TEAM-LEVEL TABLES ---
    team_trad = data.get("traditional", {}).get("TeamStats", pd.DataFrame()).copy()
    team_adv  = data.get("advanced", {}).get("TeamStats", pd.DataFrame()).copy()
    team_ff   = data.get("fourfactors", {}).get("FourFactors", pd.DataFrame()).copy()

    # --- PLAYER-LEVEL TABLES ---
    ply_trad = data.get("traditional", {}).get("PlayerStats", pd.DataFrame()).copy()
    ply_adv  = data.get("advanced", {}).get("PlayerStats", pd.DataFrame()).copy()

    # Normalize columns
    for df in [team_trad, team_adv, team_ff, ply_trad, ply_adv]:
        if not df.empty:
            df.columns = [c.upper() for c in df.columns]

    # Basic IDs / abbreviations
    for df in [team_trad, team_adv, team_ff]:
        for col in ["TEAM_ID", "TEAM_ABBREVIATION"]:
            if not df.empty and col not in df.columns and "TEAM" in df.columns:
                df.rename(columns={"TEAM": "TEAM_ABBREVIATION"}, inplace=True)

    teams = []
    if not team_trad.empty and "TEAM_ID" in team_trad.columns:
        teams = team_trad[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates().to_dict("records")
    elif not ply_trad.empty and "TEAM_ID" in ply_trad.columns:
        teams = ply_trad[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates().to_dict("records")

    if len(teams) != 2:
        # Fallback: try to infer from any available table
        if not team_adv.empty and "TEAM_ID" in team_adv.columns:
            teams = team_adv[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates().to_dict("records")
    if len(teams) != 2:
        print(f"[warn] Could not identify both teams for {game_id}")
        return []

    # Build dict for quick access by TEAM_ID
    def _row_where(df: pd.DataFrame, team_id) -> pd.Series:
        if df.empty or "TEAM_ID" not in df.columns:
            return pd.Series(dtype="object")
        match = df[df["TEAM_ID"] == team_id]
        return match.iloc[0] if not match.empty else pd.Series(dtype="object")

    # player tables split by team
    def _players_of(team_id) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pt = ply_trad[ply_trad["TEAM_ID"] == team_id].copy() if not ply_trad.empty else pd.DataFrame()
        pa = ply_adv[ply_adv["TEAM_ID"] == team_id].copy() if not ply_adv.empty else pd.DataFrame()
        return pt, pa

    rows = []
    # Create rows for each team, including opponent features
    for i, t in enumerate(teams):
        team_id = t["TEAM_ID"]
        team_abbr = t["TEAM_ABBREVIATION"]
        opp = teams[1 - i]
        opp_id, opp_abbr = opp["TEAM_ID"], opp["TEAM_ABBREVIATION"]

        tr = _row_where(team_trad, team_id)
        tr_opp = _row_where(team_trad, opp_id)

        ff = _row_where(team_ff, team_id)
        ff_opp = _row_where(team_ff, opp_id)

        ta = _row_where(team_adv, team_id)    # may be empty
        ta_opp = _row_where(team_adv, opp_id)

        pt, pa = _players_of(team_id)
        pt_opp, pa_opp = _players_of(opp_id)

        # Minutes (mp): sum player minutes (more robust)
        mp = _minutes_weighted_mean(pt.assign(MIN=pt.get("MIN")), "MIN")  # weighted mean of MIN by MIN -> returns total/players, not total
        # better: sum minutes directly
        def _sum_minutes(df):
            if df.empty or "MIN" not in df.columns:
                return None
            mins = df["MIN"].apply(_parse_min_to_float)
            return float(mins.dropna().sum()) if mins.notna().any() else None

        mp = _sum_minutes(pt)
        mp_opp = _sum_minutes(pt_opp)

        # Team FG% from traditional team stats
        fg_pct = float(tr.get("FG_PCT")) if "FG_PCT" in tr else None

        # Four Factors team percentages
        # 'ORB%','TRB%': OREB_PCT, REB_PCT from FourFactors
        orb_pct = float(ff.get("OREB_PCT")) if "OREB_PCT" in ff else None
        trb_pct = float(ff.get("REB_PCT")) if "REB_PCT" in ff else None

        # Team usage %: minutes-weighted mean of player USG_PCT
        usg_pct = _minutes_weighted_mean(pa, "USG_PCT")

        # Maxima among players on this team (traditional + advanced)
        fg_pct_max = _safe_max(pt, "FG_PCT")
        ft_max     = _safe_max(pt, "FTM")
        plus_max   = _safe_max(pt, "PLUS_MINUS")
        ts_pct_max = _safe_max(pa, "TS_PCT")
        drb_pct_max = _safe_max(pa, "DRB_PCT")
        tov_pct_max = _safe_max(pa, "TOV_PCT")
        usg_pct_max = _safe_max(pa, "USG_PCT")

        # Opponent raw counts (traditional team)
        fg_opp  = float(tr_opp.get("FGM")) if "FGM" in tr_opp else None
        three_opp = float(tr_opp.get("FG3M")) if "FG3M" in tr_opp else None
        blk_opp = float(tr_opp.get("BLK")) if "BLK" in tr_opp else None

        # Opponent team % from FourFactors / Advanced / player usage
        drb_pct_opp = float(ff_opp.get("DREB_PCT")) if "DREB_PCT" in ff_opp else None
        trb_pct_opp = float(ff_opp.get("REB_PCT")) if "REB_PCT" in ff_opp else None
        # BLK% team (advanced team stats may have BLK_PCT)
        blk_pct_opp = float(ta_opp.get("BLK_PCT")) if "BLK_PCT" in ta_opp else None
        usg_pct_opp = _minutes_weighted_mean(pa_opp, "USG_PCT")

        # Opponent maxima among opponent players
        fg_max_opp   = _safe_max(pt_opp, "FGM")
        fga_max_opp  = _safe_max(pt_opp, "FGA")
        three_pct_max_opp = _safe_max(pt_opp, "FG3_PCT")
        ft_pct_max_opp    = _safe_max(pt_opp, "FT_PCT")
        stl_max_opp  = _safe_max(pt_opp, "STL")
        blk_max_opp  = _safe_max(pt_opp, "BLK")
        pf_max_opp   = _safe_max(pt_opp, "PF")
        pts_max_opp  = _safe_max(pt_opp, "PTS")
        drb_pct_max_opp = _safe_max(pa_opp, "DRB_PCT")
        blk_pct_max_opp = _safe_max(pa_opp, "BLK_PCT")

        # Home/Away flag (try to infer from team_trad if it has MATCHUP or HOME_TEAM_ID)
        home_away = None
        if "MATCHUP" in pt.columns:
            # MATCHUP like "BOS @ ATL" or "ATL vs. BOS"
            samples = pt["MATCHUP"].dropna().astype(str)
            if not samples.empty:
                m0 = samples.iloc[0]
                if " vs. " in m0:
                    home_away = "HOME"
                elif " @ " in m0:
                    home_away = "AWAY"
        # fallback unknown
        if home_away is None:
            home_away = ""

        row = {
            # identifiers
            "GAME_ID": game_id,
            "TEAM_ID": team_id,
            "TEAM_ABBREVIATION": team_abbr,
            "OPP_TEAM_ID": opp_id,
            "OPP_TEAM_ABBREVIATION": opp_abbr,
            "HOME_AWAY": home_away,

            # requested team features
            "mp": mp,
            "fg%": fg_pct,
            "orb%": orb_pct,
            "trb%": trb_pct,
            "usg%": usg_pct,
            "fg%_max": fg_pct_max,
            "ft_max": ft_max,
            "+/-_max": plus_max,
            "ts%_max": ts_pct_max,
            "drb%_max": drb_pct_max,
            "tov%_max": tov_pct_max,
            "usg%_max": usg_pct_max,

            # opponent features (from opponent team in same game)
            "mp_opp": mp_opp,
            "fg_opp": fg_opp,
            "3p_opp": three_opp,
            "blk_opp": blk_opp,
            "drb%_opp": drb_pct_opp,
            "trb%_opp": trb_pct_opp,
            "blk%_opp": blk_pct_opp,
            "usg%_opp": usg_pct_opp,
            "fg_max_opp": fg_max_opp,
            "fga_max_opp": fga_max_opp,
            "3p%_max_opp": three_pct_max_opp,
            "ft%_max_opp": ft_pct_max_opp,
            "stl_max_opp": stl_max_opp,
            "blk_max_opp": blk_max_opp,
            "pf_max_opp": pf_max_opp,
            "pts_max_opp": pts_max_opp,
            "drb%_max_opp": drb_pct_max_opp,
            "blk%_max_opp": blk_pct_max_opp,
        }

        rows.append(row)

    return rows


# set global print options for nicer tables
pd.set_option("display.width", 160)      # max chars per line before wrapping
pd.set_option("display.max_columns", 40) # show more columns side by side
pd.set_option("display.colheader_justify", "center")

def main():
    ap = argparse.ArgumentParser(description="Extract per-team features for completed NBA games on a date.")
    ap.add_argument("--date", required=True, help="Date in MM/DD/YYYY (e.g., 01/15/2025)")
    args = ap.parse_args()

    game_ids = _final_game_ids(args.date)
    if not game_ids:
        print(f"No completed games on {args.date}")
        sys.exit(0)

    for gid in game_ids:
        rows = _team_rows_for_game(gid, args.date)
        if not rows:
            continue
        df = pd.DataFrame(rows)

        # clean up formatting
        df = df.fillna("")  # replace None with blank
        df = df.round(3)    # round floats

        teams = " vs ".join(df["TEAM_ABBREVIATION"].tolist())
        print(f"\n=== GAME_ID {gid} | {teams} ===")

        preferred = [
            "TEAM_ABBREVIATION","HOME_AWAY",
            "mp","fg%","orb%","trb%","usg%",
            "fg%_max","ft_max","+/-_max","ts%_max","drb%_max","tov%_max","usg%_max",
            "mp_opp","fg_opp","3p_opp","blk_opp","drb%_opp","trb%_opp","blk%_opp","usg%_opp",
            "fg_max_opp","fga_max_opp","3p%_max_opp","ft%_max_opp","stl_max_opp","blk_max_opp",
            "pf_max_opp","pts_max_opp","drb%_max_opp","blk%_max_opp",
        ]
        cols = [c for c in preferred if c in df.columns]

        print(df[cols].to_string(index=False))
    
if __name__ == "__main__":
    main()



# # NEXT STEPS:
# # 1. save box scores to json + confirm accuracy
# # 2. test model on old box scores
# # 3. edit script to get the last week of box scores
# # 4. UI Updates

# scripts/build_week_boxscores.py
# Usage:
#   python scripts/build_week_boxscores.py --date 01/15/2025
#
# Outputs:
#   frontend/public/data/boxscores.csv        (per-game per-team features)
#   frontend/public/data/averages.csv         (10-game averages per team)
#   frontend/public/data/scaledaverages.csv   (10-game averages, min-max scaled to [0,1])
#
# Deps:
#   pip install nba_api pandas

# scripts/build_week_boxscores.py
# Usage:
#   python scripts/build_week_boxscores.py --date 01/15/2025
#
# Outputs:
#   frontend/public/data/boxscores.csv
#   frontend/public/data/averages.csv
#   frontend/public/data/scaledaverages.csv
#
# Deps:
#   pip install nba_api pandas numpy

# from __future__ import annotations
# import argparse
# import time
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple
# import os

# import pandas as pd
# import numpy as np

# from nba_api.stats.static import teams as static_teams
# from nba_api.stats.endpoints import (
#     teamgamelog,
#     boxscoretraditionalv2,
#     boxscoreadvancedv2,
#     boxscorefourfactorsv2,
#     scoreboardv2,
#     leaguegamefinder,
# )

# # --------- Tunables (be polite) ----------
# RATE_DELAY = 1.0
# RETRY_DELAY = 3.0
# MAX_RETRIES = 3
# # ----------------------------------------

# OUTPUT_DIR = os.path.join("frontend", "public", "data")
# BOX_OUT = os.path.join(OUTPUT_DIR, "boxscores.csv")
# AVG_OUT = os.path.join(OUTPUT_DIR, "averages.csv")
# SCALED_OUT = os.path.join(OUTPUT_DIR, "scaledaverages.csv")

# # ---------- Utilities ----------
# def ensure_dir(p: str) -> None:
#     d = os.path.dirname(p)
#     if d and not os.path.isdir(d):
#         os.makedirs(d, exist_ok=True)

# def _parse_mmddyyyy(s: str) -> datetime:
#     return datetime.strptime(s, "%m/%d/%Y")

# def _season_string_for_date(dt: datetime) -> str:
#     y = dt.year
#     if dt.month >= 10:
#         return f"{y}-{str((y + 1) % 100).zfill(2)}"
#     else:
#         return f"{y - 1}-{str(y % 100).zfill(2)}"

# def _with_retries(fn, *args, **kwargs):
#     for i in range(MAX_RETRIES):
#         try:
#             return fn(*args, **kwargs)
#         except Exception as e:
#             if i == MAX_RETRIES - 1:
#                 raise
#             print(f"[warn] {fn.__name__} failed (attempt {i+1}/{MAX_RETRIES}): {e}")
#             time.sleep(RETRY_DELAY)

# def _parse_min_to_float(min_str: Optional[str]) -> Optional[float]:
#     if not isinstance(min_str, str) or ":" not in min_str:
#         return None
#     try:
#         mm, ss = min_str.split(":")
#         return int(mm) + int(ss) / 60.0
#     except Exception:
#         return None

# def _minutes_sum(df: pd.DataFrame) -> Optional[float]:
#     if df.empty or "MIN" not in df.columns:
#         return None
#     mins = df["MIN"].apply(_parse_min_to_float)
#     return float(mins.dropna().sum()) if mins.notna().any() else None

# def _minutes_weighted_mean(df: pd.DataFrame, value_col: str, min_col: str = "MIN") -> Optional[float]:
#     if df.empty or value_col not in df.columns or min_col not in df.columns:
#         return None
#     mins = df[min_col].apply(_parse_min_to_float)
#     vals = pd.to_numeric(df[value_col], errors="coerce")
#     mask = mins.notna() & vals.notna()
#     if not mask.any():
#         return None
#     w = mins[mask]
#     v = vals[mask]
#     denom = w.sum()
#     return float((v * w).sum() / denom) if denom > 0 else None

# def _safe_max(df: pd.DataFrame, col: str) -> Optional[float]:
#     if df.empty or col not in df.columns:
#         return None
#     s = pd.to_numeric(df[col], errors="coerce")
#     return float(s.max()) if s.notna().any() else None

# def _frames_by_name(endpoint_obj) -> Dict[str, pd.DataFrame]:
#     d = endpoint_obj.get_dict()
#     out = {}
#     for rs in d.get("resultSets", []):
#         name = rs.get("name", "Unknown")
#         headers = rs.get("headers", [])
#         rows = rs.get("rowSet", [])
#         out[name] = pd.DataFrame(rows, columns=headers)
#     return out

# # ---------- Box score fetch + feature extraction ----------
# _box_cache: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

# def _load_boxscore_sets(game_id: str) -> Dict[str, Dict[str, pd.DataFrame]]:
#     if game_id in _box_cache:
#         return _box_cache[game_id]

#     out = {}
#     def call_trad():
#         return boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
#     try:
#         tr = _with_retries(call_trad)
#         out["traditional"] = _frames_by_name(tr)
#     except Exception as e:
#         print(f"[warn] Traditional failed for {game_id}: {e}")
#         out["traditional"] = {}
#     time.sleep(RATE_DELAY)

#     def call_adv():
#         return boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
#     try:
#         adv = _with_retries(call_adv)
#         out["advanced"] = _frames_by_name(adv)
#     except Exception as e:
#         print(f"[warn] Advanced failed for {game_id}: {e}")
#         out["advanced"] = {}
#     time.sleep(RATE_DELAY)

#     def call_ff():
#         return boxscorefourfactorsv2.BoxScoreFourFactorsV2(game_id=game_id)
#     try:
#         ff = _with_retries(call_ff)
#         out["fourfactors"] = _frames_by_name(ff)
#     except Exception as e:
#         print(f"[warn] FourFactors failed for {game_id}: {e}")
#         out["fourfactors"] = {}

#     _box_cache[game_id] = out
#     return out

# def _team_rows_for_game(game_id: str) -> List[Dict]:
#     data = _load_boxscore_sets(game_id)

#     team_trad = data.get("traditional", {}).get("TeamStats", pd.DataFrame()).copy()
#     team_adv  = data.get("advanced", {}).get("TeamStats", pd.DataFrame()).copy()
#     team_ff   = data.get("fourfactors", {}).get("FourFactors", pd.DataFrame()).copy()

#     ply_trad  = data.get("traditional", {}).get("PlayerStats", pd.DataFrame()).copy()
#     ply_adv   = data.get("advanced", {}).get("PlayerStats", pd.DataFrame()).copy()

#     for df in [team_trad, team_adv, team_ff, ply_trad, ply_adv]:
#         if not df.empty:
#             df.columns = [c.upper() for c in df.columns]

#     teams = []
#     if not team_trad.empty and "TEAM_ID" in team_trad.columns:
#         teams = team_trad[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates().to_dict("records")
#     elif not ply_trad.empty and "TEAM_ID" in ply_trad.columns:
#         teams = ply_trad[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates().to_dict("records")

#     if len(teams) != 2 and not team_adv.empty and "TEAM_ID" in team_adv.columns:
#         teams = team_adv[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates().to_dict("records")

#     if len(teams) != 2:
#         print(f"[warn] Could not identify both teams for {game_id}")
#         return []

#     def _row_where(df: pd.DataFrame, team_id) -> pd.Series:
#         if df.empty or "TEAM_ID" not in df.columns:
#             return pd.Series(dtype="object")
#         m = df[df["TEAM_ID"] == team_id]
#         return m.iloc[0] if not m.empty else pd.Series(dtype="object")

#     def _players_of(team_id) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         pt = ply_trad[ply_trad["TEAM_ID"] == team_id].copy() if not ply_trad.empty else pd.DataFrame()
#         pa = ply_adv[ply_adv["TEAM_ID"] == team_id].copy() if not ply_adv.empty else pd.DataFrame()
#         return pt, pa

#     rows = []
#     for i, t in enumerate(teams):
#         team_id = t["TEAM_ID"]
#         abbr = t["TEAM_ABBREVIATION"]
#         opp = teams[1 - i]
#         opp_id, opp_abbr = opp["TEAM_ID"], opp["TEAM_ABBREVIATION"]

#         tr = _row_where(team_trad, team_id)
#         tr_opp = _row_where(team_trad, opp_id)
#         ff = _row_where(team_ff, team_id)
#         ff_opp = _row_where(team_ff, opp_id)
#         ta = _row_where(team_adv, team_id)
#         ta_opp = _row_where(team_adv, opp_id)

#         pt, pa = _players_of(team_id)
#         pt_opp, pa_opp = _players_of(opp_id)

#         mp = _minutes_sum(pt)
#         mp_opp = _minutes_sum(pt_opp)

#         fg_pct = float(tr.get("FG_PCT")) if "FG_PCT" in tr else None

#         orb_pct = float(ff.get("OREB_PCT")) if "OREB_PCT" in ff else None
#         trb_pct = float(ff.get("REB_PCT")) if "REB_PCT" in ff else None

#         usg_pct = _minutes_weighted_mean(pa, "USG_PCT")
#         usg_pct_opp = _minutes_weighted_mean(pa_opp, "USG_PCT")

#         fg_pct_max = _safe_max(pt, "FG_PCT")
#         ft_max     = _safe_max(pt, "FTM")
#         plus_max   = _safe_max(pt, "PLUS_MINUS")
#         ts_pct_max = _safe_max(pa, "TS_PCT")
#         drb_pct_max = _safe_max(pa, "DRB_PCT")
#         tov_pct_max = _safe_max(pa, "TOV_PCT")
#         usg_pct_max = _safe_max(pa, "USG_PCT")

#         fg_opp  = float(tr_opp.get("FGM")) if "FGM" in tr_opp else None
#         three_opp = float(tr_opp.get("FG3M")) if "FG3M" in tr_opp else None
#         blk_opp = float(tr_opp.get("BLK")) if "BLK" in tr_opp else None

#         drb_pct_opp = float(ff_opp.get("DREB_PCT")) if "DREB_PCT" in ff_opp else None
#         trb_pct_opp = float(ff_opp.get("REB_PCT")) if "REB_PCT" in ff_opp else None
#         blk_pct_opp = float(ta_opp.get("BLK_PCT")) if "BLK_PCT" in ta_opp else None

#         fg_max_opp   = _safe_max(pt_opp, "FGM")
#         fga_max_opp  = _safe_max(pt_opp, "FGA")
#         three_pct_max_opp = _safe_max(pt_opp, "FG3_PCT")
#         ft_pct_max_opp    = _safe_max(pt_opp, "FT_PCT")
#         stl_max_opp  = _safe_max(pt_opp, "STL")
#         blk_max_opp  = _safe_max(pt_opp, "BLK")
#         pf_max_opp   = _safe_max(pt_opp, "PF")
#         pts_max_opp  = _safe_max(pt_opp, "PTS")
#         drb_pct_max_opp = _safe_max(pa_opp, "DRB_PCT")
#         blk_pct_max_opp = _safe_max(pa_opp, "BLK_PCT")

#         row = {
#             "GAME_ID": str(game_id),
#             "TEAM_ID": int(team_id),
#             "TEAM_ABBREVIATION": abbr,
#             "OPP_TEAM_ID": int(opp_id),
#             "OPP_TEAM_ABBREVIATION": opp_abbr,

#             "mp": mp,
#             "fg%": fg_pct,
#             "orb%": orb_pct,
#             "trb%": trb_pct,
#             "usg%": usg_pct,
#             "fg%_max": fg_pct_max,
#             "ft_max": ft_max,
#             "+/-_max": plus_max,
#             "ts%_max": ts_pct_max,
#             "drb%_max": drb_pct_max,
#             "tov%_max": tov_pct_max,
#             "usg%_max": usg_pct_max,

#             "mp_opp": mp_opp,
#             "fg_opp": fg_opp,
#             "3p_opp": three_opp,
#             "blk_opp": blk_opp,
#             "drb%_opp": drb_pct_opp,
#             "trb%_opp": trb_pct_opp,
#             "blk%_opp": blk_pct_opp,
#             "usg%_opp": usg_pct_opp,

#             "fg_max_opp": fg_max_opp,
#             "fga_max_opp": fga_max_opp,
#             "3p%_max_opp": three_pct_max_opp,
#             "ft%_max_opp": ft_pct_max_opp,
#             "stl_max_opp": stl_max_opp,
#             "blk_max_opp": blk_max_opp,
#             "pf_max_opp": pf_max_opp,
#             "pts_max_opp": pts_max_opp,
#             "drb%_max_opp": drb_pct_max_opp,
#             "blk%_max_opp": blk_pct_max_opp,
#         }
#         rows.append(row)

#     return rows

# # ---------- Scaling ----------
# def min_max_scale(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
#     out = df.copy()
#     for col in out.columns:
#         if col in exclude:
#             continue
#         if pd.api.types.is_numeric_dtype(out[col]):
#             col_min = out[col].min()
#             col_max = out[col].max()
#             if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
#                 out[col] = 0.0
#             else:
#                 out[col] = (out[col] - col_min) / (col_max - col_min)
#     return out

# # ---------- One-team helpers ----------
# def _resolve_team_id(team_arg: str) -> Tuple[int, str]:
#     all_teams = static_teams.get_teams()
#     by_abbr = {t["abbreviation"].upper(): (t["id"], t["abbreviation"]) for t in all_teams}
#     by_id = {int(t["id"]): t["abbreviation"] for t in all_teams}
#     s = team_arg.strip().upper()
#     if s.isdigit():
#         tid = int(s)
#         if tid not in by_id:
#             raise ValueError(f"Unknown team id: {tid}")
#         return tid, by_id[tid]
#     if s not in by_abbr:
#         raise ValueError(f"Unknown team abbreviation: {s}")
#     tid, abbr = by_abbr[s]
#     return tid, abbr

# from nba_api.stats.endpoints import scoreboardv2

# def get_last_n_game_ids_for_team_one(team_id: int, cutoff_mmddyyyy: str, n: int = 10, sleep_s: float = RATE_DELAY) -> List[str]:
#     """Return the last N game_ids for a team on/before cutoff date."""
#     cutoff_dt = _parse_mmddyyyy(cutoff_mmddyyyy).date()
#     found: List[str] = []

#     # --- Primary path: LeagueGameFinder (fast; returns played games) ---
#     try:
#         lgf = leaguegamefinder.LeagueGameFinder(
#             team_id_nullable=team_id,
#             season_type_nullable="Regular Season",
#         )
#         gdf = lgf.get_data_frames()[0].copy()
#         if not gdf.empty and "GAME_DATE" in gdf.columns and "GAME_ID" in gdf.columns:
#             gdf["GAME_DATE"] = pd.to_datetime(gdf["GAME_DATE"], errors="coerce").dt.date
#             # Keep games on/before cutoff
#             gdf = gdf[gdf["GAME_DATE"] <= cutoff_dt]
#             # Sort newest → oldest and take last N
#             gdf = gdf.sort_values("GAME_DATE", ascending=False)
#             found = gdf["GAME_ID"].astype(str).head(n).tolist()
#             if found:
#                 return found
#     except Exception as e:
#         print(f"[warn] LeagueGameFinder path failed: {e}")

#     # --- Fallback: walk days backward with ScoreboardV2 (slower, but robust) ---
#     try:
#         for offset in range(45):  # ~1.5 months back just in case
#             day = (cutoff_dt - pd.Timedelta(days=offset)).strftime("%m/%d/%Y")
#             try:
#                 sb = scoreboardv2.ScoreboardV2(game_date=day)
#                 frames = sb.get_data_frames()
#                 if not frames:
#                     continue
#                 gh = frames[0].copy()  # GameHeader
#                 gh.columns = [c.upper() for c in gh.columns]
#                 finals = gh[gh["GAME_STATUS_TEXT"].astype(str).str.lower() == "final"]
#                 team_games = finals[(finals["HOME_TEAM_ID"] == team_id) | (finals["VISITOR_TEAM_ID"] == team_id)]
#                 for gid in team_games["GAME_ID"].astype(str).tolist():
#                     if gid not in found:
#                         found.append(gid)
#                 if len(found) >= n:
#                     break
#             except Exception as e:
#                 print(f"[warn] scoreboard fallback failed for {day}: {e}")
#             time.sleep(sleep_s)
#     except Exception as e:
#         print(f"[warn] fallback loop failed: {e}")

#     return found[:n]

# # ---------- Main (one team) ----------
# def main():
#     ap = argparse.ArgumentParser(description="Build last-10 box score features for ONE team before a cutoff date.")
#     ap.add_argument("--team", required=True, help="Team abbreviation (e.g., BOS) or numeric team ID")
#     ap.add_argument("--date", required=True, help="Cutoff date in MM/DD/YYYY (e.g., 01/15/2025)")
#     args = ap.parse_args()

#     ensure_dir(BOX_OUT)
#     ensure_dir(AVG_OUT)
#     ensure_dir(SCALED_OUT)

#     team_id, team_abbr = _resolve_team_id(args.team)
#     print(f"[info] Target team: {team_abbr} (id={team_id}), cutoff {args.date}")

#     game_ids = get_last_n_game_ids_for_team_one(team_id, args.date, n=10, sleep_s=RATE_DELAY)
#     print(f"[info] {team_abbr} last10 (<= {args.date}): {len(game_ids)} games")
#     if not game_ids:
#         print("[error] No games found. Check the cutoff date/season and connectivity to stats.nba.com")
#         return

#     # Fetch once per game, build per-team rows then keep only this team
#     rows: List[Dict] = []
#     for gid in game_ids:
#         try:
#             print(f"[info] Fetching box score for {gid} …")
#             rows.extend(_team_rows_for_game(gid))
#         except Exception as e:
#             print(f"[warn] Skipping {gid}: {e}")

#     if not rows:
#         print("[error] No feature rows assembled.")
#         return

#     df = pd.DataFrame(rows)
#     df = df[df["TEAM_ID"] == team_id].copy()

#     df_sorted = df.sort_values("GAME_ID").reset_index(drop=True)
#     df_sorted.to_csv(BOX_OUT, index=False)
#     print(f"[ok] wrote {BOX_OUT} ({len(df_sorted)} rows)")

#     # Averages for this one team
#     id_cols = ["TEAM_ID", "TEAM_ABBREVIATION"]
#     num_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
#     avg = df.groupby(id_cols, as_index=False)[num_cols].mean()
#     avg["games_count"] = len(df_sorted)
#     avg.to_csv(AVG_OUT, index=False)
#     print(f"[ok] wrote {AVG_OUT}")

#     scaled = min_max_scale(avg, exclude=id_cols + ["games_count"])
#     scaled.to_csv(SCALED_OUT, index=False)
#     print(f"[ok] wrote {SCALED_OUT}")

# if __name__ == "__main__":
#     main()
