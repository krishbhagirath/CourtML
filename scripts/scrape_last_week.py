# scripts/scrape_last_week.py
# Usage:
#   Live (default):    python scripts/scrape_last_week.py
#   Offline test:      python scripts/scrape_last_week.py --html-file "C:\path\to\NBA_2026_games.html"
#
# What it does:
# - LAST 7 FULL DAYS ending yesterday in America/Toronto.
# - If that window is empty (offseason/early pre-season), it loads the PREVIOUS season page
#   and returns the final 7 calendar days of that season (ending on the last game date).
#   If >10 games fall in that 7-day window, it trims to the LAST 10 games.
# - Saves UI-shaped JSON (same as your site) to: ../frontend/public/data/previous.json
# - Includes final scores; prediction is None.

import argparse
import io
import json
import random
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# ----------------------------
# Config
# ----------------------------
LOCAL_TZ = ZoneInfo("America/Toronto")
SEASON_ENDING_YEAR = 2026  # NBA_2026_games.html (2025-26 season)
BASE_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_games.html"
SEASON_URL = BASE_URL.format(year=SEASON_ENDING_YEAR)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = (SCRIPT_DIR.parent / "frontend" / "public" / "data" / "lastweek.json")

# Polite delays
INITIAL_SLEEP_RANGE = (6, 12)  # seconds before first request
RETRY_BACKOFF_BASE = 8.0       # base seconds between retries
MAX_RETRIES = 4

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
}

# ----------------------------
# Team mapping (BBR name → {name, abbr, city})
# ----------------------------
TEAMS = {
    "Atlanta Hawks":              {"name": "Hawks", "abbreviation": "ATL", "city": "Atlanta"},
    "Boston Celtics":             {"name": "Celtics", "abbreviation": "BOS", "city": "Boston"},
    "Brooklyn Nets":              {"name": "Nets", "abbreviation": "BKN", "city": "Brooklyn"},
    "Charlotte Hornets":          {"name": "Hornets", "abbreviation": "CHA", "city": "Charlotte"},
    "Chicago Bulls":              {"name": "Bulls", "abbreviation": "CHI", "city": "Chicago"},
    "Cleveland Cavaliers":        {"name": "Cavaliers", "abbreviation": "CLE", "city": "Cleveland"},
    "Dallas Mavericks":           {"name": "Mavericks", "abbreviation": "DAL", "city": "Dallas"},
    "Denver Nuggets":             {"name": "Nuggets", "abbreviation": "DEN", "city": "Denver"},
    "Detroit Pistons":            {"name": "Pistons", "abbreviation": "DET", "city": "Detroit"},
    "Golden State Warriors":      {"name": "Warriors", "abbreviation": "GSW", "city": "Golden State"},
    "Houston Rockets":            {"name": "Rockets", "abbreviation": "HOU", "city": "Houston"},
    "Indiana Pacers":             {"name": "Pacers", "abbreviation": "IND", "city": "Indiana"},
    "LA Clippers":                {"name": "Clippers", "abbreviation": "LAC", "city": "Los Angeles"},
    "Los Angeles Clippers":       {"name": "Clippers", "abbreviation": "LAC", "city": "Los Angeles"},
    "Los Angeles Lakers":         {"name": "Lakers", "abbreviation": "LAL", "city": "Los Angeles"},
    "Memphis Grizzlies":          {"name": "Grizzlies", "abbreviation": "MEM", "city": "Memphis"},
    "Miami Heat":                 {"name": "Heat", "abbreviation": "MIA", "city": "Miami"},
    "Milwaukee Bucks":            {"name": "Bucks", "abbreviation": "MIL", "city": "Milwaukee"},
    "Minnesota Timberwolves":     {"name": "Timberwolves", "abbreviation": "MIN", "city": "Minnesota"},
    "New Orleans Pelicans":       {"name": "Pelicans", "abbreviation": "NOP", "city": "New Orleans"},
    "New York Knicks":            {"name": "Knicks", "abbreviation": "NYK", "city": "New York"},
    "Oklahoma City Thunder":      {"name": "Thunder", "abbreviation": "OKC", "city": "Oklahoma City"},
    "Orlando Magic":              {"name": "Magic", "abbreviation": "ORL", "city": "Orlando"},
    "Philadelphia 76ers":         {"name": "76ers", "abbreviation": "PHI", "city": "Philadelphia"},
    "Phoenix Suns":               {"name": "Suns", "abbreviation": "PHX", "city": "Phoenix"},
    "Portland Trail Blazers":     {"name": "Trail Blazers", "abbreviation": "POR", "city": "Portland"},
    "Sacramento Kings":           {"name": "Kings", "abbreviation": "SAC", "city": "Sacramento"},
    "San Antonio Spurs":          {"name": "Spurs", "abbreviation": "SAS", "city": "San Antonio"},
    "Toronto Raptors":            {"name": "Raptors", "abbreviation": "TOR", "city": "Toronto"},
    "Utah Jazz":                  {"name": "Jazz", "abbreviation": "UTA", "city": "Utah"},
    "Washington Wizards":         {"name": "Wizards", "abbreviation": "WAS", "city": "Washington"},
}

# ----------------------------
# Fetch (very polite)
# ----------------------------
def fetch_html(url: str) -> str:
    s = requests.Session()
    s.headers.update(HEADERS)

    # Initial wait
    time.sleep(random.uniform(*INITIAL_SLEEP_RANGE))

    # Warm-up homepage to collect benign cookies
    try:
        s.get("https://www.basketball-reference.com/", timeout=30, allow_redirects=True)
        time.sleep(random.uniform(1.0, 2.0))
    except requests.RequestException:
        pass

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(random.uniform(0.5, 1.5))
            resp = s.get(url, timeout=30, allow_redirects=True)
            if resp.status_code in (403, 429, 503):
                last_err = requests.HTTPError(f"{resp.status_code} for {url}")
                time.sleep(RETRY_BACKOFF_BASE * attempt + random.uniform(4, 8))
                continue
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            last_err = e
            time.sleep(RETRY_BACKOFF_BASE * attempt + random.uniform(4, 8))
            continue

    raise last_err or RuntimeError("Failed to fetch HTML")

# ----------------------------
# Parse schedule tables
# ----------------------------
def parse_schedule_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))

    # pick monthly schedule tables
    sched_like = []
    for t in tables:
        cols = [str(c).lower() for c in t.columns.tolist()]
        if "date" in cols and any("visitor" in c for c in cols) and any("home" in c for c in cols):
            sched_like.append(t)
    if not sched_like:
        raise RuntimeError("Could not find schedule tables on the page.")

    df = pd.concat(sched_like, ignore_index=True)

    # robust column mapping (same pattern as your script)
    colmap, visitor_pts_seen = {}, False
    for c in df.columns:
        lc = str(c).lower()
        if lc.startswith("date"):
            colmap[c] = "date"
        elif "start" in lc:
            colmap[c] = "start_et"
        elif "visitor" in lc:
            colmap[c] = "visitor"
        elif "home" in lc:
            colmap[c] = "home"
        elif lc == "pts" and not visitor_pts_seen:
            colmap[c] = "visitor_pts"; visitor_pts_seen = True
        elif lc == "pts" and visitor_pts_seen:
            colmap[c] = "home_pts"
        elif "arena" in lc:
            colmap[c] = "arena"
        elif "notes" in lc:
            colmap[c] = "notes"
        elif "attend" in lc:
            colmap[c] = "attendance"
    df = df.rename(columns=colmap)

    # remove repeated header rows; keep rows with real dates like "Tue, Oct 21, 2025"
    df = df[df["date"].astype(str).str.contains(",")].copy()

    # parse dates
    df["game_date"] = df["date"].apply(lambda s: datetime.strptime(s.strip(), "%a, %b %d, %Y").date())

    # clean strings
    for col in ["start_et","visitor","home","arena","notes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ensure score columns exist
    for col in ["visitor_pts", "home_pts"]:
        if col not in df.columns:
            df[col] = None

    return df

def fetch_schedule_df_live(url: str) -> pd.DataFrame:
    html = fetch_html(url)
    return parse_schedule_html(html)

# ----------------------------
# Time normalization (for sorting)
# ----------------------------
def time_to_minutes(t) -> int:
    if t is None:
        return 10**9
    s = str(t).strip().lower().replace(" ", "")
    if s in ("", "nan", "tbd", "ppd"):
        return 10**9
    if s.endswith("et"):
        s = s[:-2]
    ampm = "a" if "am" in s or s.endswith("a") else "p" if "pm" in s or s.endswith("p") else None
    s = s.replace("am", "").replace("pm", "").rstrip("ap")
    if ":" not in s:
        return 10**9
    hh_str, mm_str = s.split(":", 1)
    try:
        hh = int(hh_str)
        mm = int(mm_str[:2])
    except ValueError:
        return 10**9
    hh = hh % 12
    if ampm == "p":
        hh += 12
    return hh * 60 + mm

# ----------------------------
# Window helpers (LAST 7 FULL DAYS)
# ----------------------------
def last_7_full_days(df: pd.DataFrame, today: date) -> tuple[pd.DataFrame, date]:
    """
    Return (window_df, start_date) for the last 7 COMPLETE days:
      end = yesterday, start = end - 6 days
    """
    end = today - timedelta(days=1)
    start = end - timedelta(days=6)
    window = df[(df["game_date"] >= start) & (df["game_date"] <= end)].copy()
    return window, start

def previous_season_final_week(df_prev: pd.DataFrame) -> tuple[pd.DataFrame, date]:
    """
    Use the final 7 calendar days of the previous season (ending on max date).
    If >10 games fall into that 7-day window, trim to the last 10 by date/time.
    """
    last_date = df_prev["game_date"].max()
    start = last_date - timedelta(days=6)
    window = df_prev[(df_prev["game_date"] >= start) & (df_prev["game_date"] <= last_date)].copy()

    # Sort by datetime-like key within date to preserve end-of-season order
    window["__tip"] = window["start_et"].apply(time_to_minutes)
    window = window.sort_values(["game_date", "__tip"], kind="mergesort")

    # If more than 10 games, keep the last 10 (closest to season end)
    if len(window) > 10:
        window = window.tail(10)
        start = window["game_date"].min()

    return window, start

def weekday_name(d: date) -> str:
    return d.strftime("%A")

# ----------------------------
# Build UI JSON (last-week)
# ----------------------------
def to_ui_json_lastweek(window: pd.DataFrame, start: date) -> dict:
    tmp = window.copy()
    tmp["__tip"] = tmp["start_et"].apply(time_to_minutes)
    tmp = tmp.sort_values(["game_date", "__tip"], kind="mergesort")

    by_date = {}
    for d, chunk in tmp.groupby("game_date"):
        games = []
        gid = 1
        for _, r in chunk.iterrows():
            home = r.get("home")
            away = r.get("visitor")
            arena = r.get("arena") or None

            home_map = TEAMS.get(home, {
                "name": (home.split()[-1] if home else None),
                "abbreviation": None,
                "city": (home.split()[0] if home else None)
            })
            away_map = TEAMS.get(away, {
                "name": (away.split()[-1] if away else None),
                "abbreviation": None,
                "city": (away.split()[0] if away else None)
            })

            # parse scores as ints when possible
            def as_int(x):
                try:
                    return int(x)
                except (TypeError, ValueError):
                    return None

            games.append({
                "id": gid,
                "homeTeam": {
                    "name": home_map["name"],
                    "abbreviation": home_map["abbreviation"],
                    "city": home_map["city"]
                },
                "awayTeam": {
                    "name": away_map["name"],
                    "abbreviation": away_map["abbreviation"],
                    "city": away_map["city"]
                },
                "time": None,                # historical view: UI can ignore; tip time not essential
                "venue": arena,
                "date": d.isoformat(),
                "prediction": None,          # per request
                "confidence": 0,
                "gameStatus": "final",
                "finalScore": {
                    "home": as_int(r.get("home_pts")),
                    "away": as_int(r.get("visitor_pts")),
                }
            })
            gid += 1
        by_date[d] = games

    # Exactly seven consecutive day keys (even if some empty) starting at 'start'
    current_week = {}
    ordered_days = []
    week_dates_iso = []
    for i in range(7):
        this_day = start + timedelta(days=i)
        label = weekday_name(this_day)
        current_week[label] = by_date.get(this_day, [])
        ordered_days.append(label)
        week_dates_iso.append(this_day.isoformat())

    total_games = sum(len(v) for v in current_week.values())
    payload = {
        "currentWeek": current_week,
        "orderedDays": ordered_days,
        "weekDates": week_dates_iso,
        "metadata": {
            "lastUpdated": datetime.now(tz=LOCAL_TZ).isoformat(),
            "weekStartDate": week_dates_iso[0],
            "weekEndDate": week_dates_iso[-1],
            "totalGames": total_games,
            "dataSource": "Basketball-Reference (parsed)",
            "version": "1.0",
            "mode": "last-week"
        }
    }
    return payload

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build last-week NBA results JSON for the site.")
    ap.add_argument("--html-file", help="Parse from a local saved HTML file instead of fetching live.")
    args = ap.parse_args()

    today = datetime.now(tz=LOCAL_TZ).date()

    # 1) Fetch + parse
    if args.html_file:
        print(f"[main] Parsing local file: {args.html_file}")
        with open(args.html_file, "r", encoding="utf-8") as f:
            html = f.read()
        df_current = parse_schedule_html(html)
    else:
        df_current = fetch_schedule_df_live(SEASON_URL)

    # 2) Try LAST 7 FULL DAYS on the current season page
    window, start = last_7_full_days(df_current, today)

    # 3) If empty, treat as offseason: load PREVIOUS season and take final week
    if window.empty:
        prev_year = SEASON_ENDING_YEAR - 1
        prev_url = BASE_URL.format(year=prev_year)
        print("[main] No games in last-7-day window on current season page; using previous season final week.")
        if args.html_file:
            # If testing offline and only one file provided, just reuse the same parsed df.
            # You can supply another --html-file for last season if desired.
            df_prev = df_current
        else:
            df_prev = fetch_schedule_df_live(prev_url)

        window, start = previous_season_final_week(df_prev)

    # 4) Build UI-shaped JSON
    payload = to_ui_json_lastweek(window, start)

    # 5) Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 6) Console confirmation
    print(f"Saved {payload['metadata']['totalGames']} games "
          f"for {payload['metadata']['weekStartDate']} → {payload['metadata']['weekEndDate']}")
    print(f"→ {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
