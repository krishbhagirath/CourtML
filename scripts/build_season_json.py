# scripts/build_season_full.py
# Merge all monthly CSVs in data/2026_schedule into one full-season JSON:
# frontend/public/data/season.json
#
# Usage:
#   python scripts/build_season_full.py
#   python scripts/build_season_full.py --in-dir data/2026_schedule --out frontend/public/data/season.json
#
# Deps: pip install pandas python-dateutil

import argparse, glob, os, json, re
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
from dateutil import tz

# --- TEAM NAME -> (city, name, abbr) ---
TEAM_MAP: Dict[str, Dict[str, str]] = {
    "Atlanta Hawks": {"city": "Atlanta", "name": "Hawks", "abbr": "ATL"},
    "Boston Celtics": {"city": "Boston", "name": "Celtics", "abbr": "BOS"},
    "Brooklyn Nets": {"city": "Brooklyn", "name": "Nets", "abbr": "BKN"},
    "Charlotte Hornets": {"city": "Charlotte", "name": "Hornets", "abbr": "CHA"},
    "Chicago Bulls": {"city": "Chicago", "name": "Bulls", "abbr": "CHI"},
    "Cleveland Cavaliers": {"city": "Cleveland", "name": "Cavaliers", "abbr": "CLE"},
    "Dallas Mavericks": {"city": "Dallas", "name": "Mavericks", "abbr": "DAL"},
    "Denver Nuggets": {"city": "Denver", "name": "Nuggets", "abbr": "DEN"},
    "Detroit Pistons": {"city": "Detroit", "name": "Pistons", "abbr": "DET"},
    "Golden State Warriors": {"city": "Golden State", "name": "Warriors", "abbr": "GSW"},
    "Houston Rockets": {"city": "Houston", "name": "Rockets", "abbr": "HOU"},
    "Indiana Pacers": {"city": "Indiana", "name": "Pacers", "abbr": "IND"},
    "Los Angeles Clippers": {"city": "Los Angeles", "name": "Clippers", "abbr": "LAC"},
    "Los Angeles Lakers": {"city": "Los Angeles", "name": "Lakers", "abbr": "LAL"},
    "Memphis Grizzlies": {"city": "Memphis", "name": "Grizzlies", "abbr": "MEM"},
    "Miami Heat": {"city": "Miami", "name": "Heat", "abbr": "MIA"},
    "Milwaukee Bucks": {"city": "Milwaukee", "name": "Bucks", "abbr": "MIL"},
    "Minnesota Timberwolves": {"city": "Minnesota", "name": "Timberwolves", "abbr": "MIN"},
    "New Orleans Pelicans": {"city": "New Orleans", "name": "Pelicans", "abbr": "NOP"},
    "New York Knicks": {"city": "New York", "name": "Knicks", "abbr": "NYK"},
    "Oklahoma City Thunder": {"city": "Oklahoma City", "name": "Thunder", "abbr": "OKC"},
    "Orlando Magic": {"city": "Orlando", "name": "Magic", "abbr": "ORL"},
    "Philadelphia 76ers": {"city": "Philadelphia", "name": "76ers", "abbr": "PHI"},
    "Phoenix Suns": {"city": "Phoenix", "name": "Suns", "abbr": "PHX"},
    "Portland Trail Blazers": {"city": "Portland", "name": "Trail Blazers", "abbr": "POR"},
    "Sacramento Kings": {"city": "Sacramento", "name": "Kings", "abbr": "SAC"},
    "San Antonio Spurs": {"city": "San Antonio", "name": "Spurs", "abbr": "SAS"},
    "Toronto Raptors": {"city": "Toronto", "name": "Raptors", "abbr": "TOR"},
    "Utah Jazz": {"city": "Utah", "name": "Jazz", "abbr": "UTA"},
    "Washington Wizards": {"city": "Washington", "name": "Wizards", "abbr": "WAS"},
}

def normcol(c: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(c).strip()).strip("_").lower()

def sstrip(x) -> str:
    return x.strip() if isinstance(x, str) else ""

def parse_br_date(s: str) -> Optional[str]:
    # "Wed Oct 22 2025" -> "2025-10-22"
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        dt = datetime.strptime(s.strip(), "%a %b %d %Y")
    except ValueError:
        return None
    return dt.date().isoformat()

def parse_time_et_label(raw) -> Optional[str]:
    # keep "7:30p" / "10:00p" if matches; else cleaned string or None
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip().lower()
    if re.match(r"^\d{1,2}:\d{2}[ap]$", s):
        h, m_ap = s.split(":")
        m = m_ap[:2]
        ap = m_ap[2]
        return f"{int(h)}:{m}{ap}"
    return raw.strip()

def coalesce_pts_columns(df: pd.DataFrame):
    pts_cols = [c for c in df.columns if normcol(c) == "pts" or normcol(c).startswith("pts")]
    pts_cols = list(dict.fromkeys(pts_cols))  # preserve order
    v_col = pts_cols[0] if len(pts_cols) >= 1 else None
    h_col = pts_cols[1] if len(pts_cols) >= 2 else None
    return v_col, h_col

def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    want = [normcol(c) for c in candidates]
    for c in df.columns:
        if normcol(c) in want:
            return c
    return None

def load_month(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python")

def to_team_obj(br_full: str) -> Dict[str, str]:
    info = TEAM_MAP.get(br_full)
    if info:
        return {"name": info["name"], "abbreviation": info["abbr"], "city": info["city"]}
    parts = (br_full or "").split()
    name = parts[-1] if parts else br_full
    city = " ".join(parts[:-1]) if len(parts) > 1 else br_full
    abbr = "".join(w[0].upper() for w in parts[:3])[:3]
    return {"name": name, "abbreviation": abbr, "city": city}

def build_long_df(in_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    if not paths:
        raise SystemExit(f"No CSV files in {in_dir}")
    frames = []
    for p in paths:
        mdf = load_month(p)
        col_date  = first_present(mdf, ["Date"])
        col_time  = first_present(mdf, ["Start (ET)", "Start(ET)", "Start"])
        col_vis   = first_present(mdf, ["Visitor/Neutral","Visitor"])
        col_home  = first_present(mdf, ["Home/Neutral","Home"])
        col_arena = first_present(mdf, ["Arena"])
        v_pts_col, h_pts_col = coalesce_pts_columns(mdf)

        out_rows = []
        for _, r in mdf.iterrows():
            date_iso = parse_br_date(r.get(col_date)) if col_date else None
            if not date_iso:  # skip blank/headers
                continue
            vis = sstrip(r.get(col_vis)) if col_vis else ""
            home = sstrip(r.get(col_home)) if col_home else ""
            if not vis and not home:
                continue
            time_label = parse_time_et_label(r.get(col_time)) if col_time else None
            arena = sstrip(r.get(col_arena)) if col_arena else None

            def to_int(x):
                try:
                    if pd.isna(x):
                        return None
                except Exception:
                    pass
                try:
                    return int(str(x).replace(",", "").strip())
                except Exception:
                    return None
            visitor_pts = to_int(r.get(v_pts_col)) if v_pts_col else None
            home_pts    = to_int(r.get(h_pts_col)) if h_pts_col else None

            out_rows.append({
                "date": date_iso,
                "time": time_label or "",
                "visitor": vis,
                "visitor_pts": visitor_pts,
                "home": home,
                "home_pts": home_pts,
                "venue": arena or "",
            })
        frames.append(pd.DataFrame(out_rows))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df["__ds"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["__ds","time","home","visitor"]).drop(columns="__ds").reset_index(drop=True)
    return df

def build_season_payload(df: pd.DataFrame) -> Dict:
    # group into gamesByDate; assign per-date 1..N ids like your week example
    games_by_date: Dict[str, List[Dict]] = {}
    season_dates = sorted(df["date"].dropna().unique().tolist())

    for d in season_dates:
        day_df = df[df["date"] == d].reset_index(drop=True)
        rows = []
        for i, r in enumerate(day_df.itertuples(index=False), start=1):
            home_obj = to_team_obj(r.home)
            away_obj = to_team_obj(r.visitor)
            rows.append({
                "id": i,
                "homeTeam": home_obj,
                "awayTeam": away_obj,
                "time": r.time,
                "venue": r.venue,
                "date": r.date,
                "prediction": {"winner": "TBD", "confidence": 0, "spread": None, "overUnder": None},
                "confidence": 0,
                "gameStatus": "scheduled"
            })
        games_by_date[d] = rows

    payload = {
        "seasonDates": season_dates,
        "gamesByDate": games_by_date,
        "metadata": {
            "lastUpdated": datetime.now(tz.gettz("America/Toronto")).isoformat(),
            "totalGames": int(sum(len(g) for g in games_by_date.values())),
            "dataSource": "Basketball-Reference (parsed)",
            "version": "1.0"
        }
    }
    return payload

def main():
    ap = argparse.ArgumentParser(description="Build full-season JSON from monthly Basketball-Reference CSVs.")
    ap.add_argument("--in-dir", default="data/2026_schedule", help="Folder with month CSVs (october.csv..april.csv)")
    ap.add_argument("--out", default="frontend/public/data/season.json", help="Output JSON path")
    args = ap.parse_args()

    df = build_long_df(args.in_dir)
    if df.empty:
        raise SystemExit("No rows parsed. Check your CSVs in data/2026_schedule.")

    payload = build_season_payload(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[ok] Wrote {args.out} with {payload['metadata']['totalGames']} games across {len(payload['seasonDates'])} dates.")

if __name__ == "__main__":
    main()
