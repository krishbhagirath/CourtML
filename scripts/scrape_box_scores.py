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



# NEXT STEPS:
# 1. save box scores to json + confirm accuracy
# 2. test model on old box scores
# 3. edit script to get the last week of box scores
# 4. UI Updates