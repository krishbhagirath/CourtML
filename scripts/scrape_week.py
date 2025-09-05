#!/usr/bin/env python3
# scripts/scrape_week.py
#
# Usage:
#   python scripts/scrape_week.py                   # use today's date
#   python scripts/scrape_week.py --start-date 2025-11-03
#   python scripts/scrape_week.py --season-year 2026
#   python scripts/scrape_week.py --min-delay 10 --max-delay 20 --between-pages 8

import argparse
import io
import json
import random
import re
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

LOCAL_TZ = ZoneInfo("America/Toronto")
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "frontend" / "public" / "data" / "upcoming.json"

# polite defaults
DEFAULT_MIN_DELAY = 12
DEFAULT_MAX_DELAY = 22
DEFAULT_BETWEEN_PAGES = 10

HEADERS = {"User-Agent": "Mozilla/5.0"}

def season_index_url(year: int) -> str:
    return f"https://www.basketball-reference.com/leagues/NBA_{year}_games.html"

def season_month_url(year: int, month: str) -> str:
    return f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"

def fetch_html(url: str, min_delay: int, max_delay: int) -> str:
    time.sleep(random.uniform(min_delay, max_delay))
    for attempt in range(4):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code in (403, 429, 503):
                time.sleep((15 * (attempt+1)) + random.uniform(5,10))
                continue
            resp.raise_for_status()
            return resp.text
        except Exception:
            time.sleep((15 * (attempt+1)) + random.uniform(5,10))
    raise RuntimeError(f"Failed to fetch {url}")

def parse_schedule_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    sched = [t for t in tables if "Date" in t.columns and "Visitor/Neutral" in t.columns]
    df = pd.concat(sched, ignore_index=True)
    df = df[df["Date"].astype(str).str.contains(",")]
    df["game_date"] = pd.to_datetime(df["Date"]).dt.date
    return df.rename(columns={"Visitor/Neutral":"visitor","Home/Neutral":"home",
                              "PTS":"visitor_pts","PTS.1":"home_pts","Start (ET)":"start_et"})

def discover_month_slugs(index_html: str, season_year: int):
    return re.findall(rf"NBA_{season_year}_games-([a-z]+)\.html", index_html)

def fetch_full_season_df(season_year: int, min_delay: int, max_delay: int, between_pages: int) -> pd.DataFrame:
    idx_html = fetch_html(season_index_url(season_year), min_delay, max_delay)
    months = discover_month_slugs(idx_html, season_year)
    frames = [parse_schedule_html(idx_html)]
    for m in months:
        html = fetch_html(season_month_url(season_year, m), min_delay, max_delay)
        frames.append(parse_schedule_html(html))
        time.sleep(random.uniform(between_pages, between_pages+5))
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["game_date","visitor","home"])

def window_next7(df: pd.DataFrame, anchor: date):
    window = df[(df["game_date"]>=anchor) & (df["game_date"]<=anchor+timedelta(days=6))]
    if window.empty:
        start = df["game_date"].min()
        return df[(df["game_date"]>=start) & (df["game_date"]<=start+timedelta(days=6))], start
    return window, anchor

def to_ui_json(window: pd.DataFrame, start: date):
    current_week = {}
    for i in range(7):
        d = start + timedelta(days=i)
        games = []
        chunk = window[window["game_date"]==d]
        for gid, r in enumerate(chunk.to_dict("records"),1):
            games.append({
                "id": gid,
                "homeTeam": {"name": r["home"].split()[-1], "abbreviation": None, "city": r["home"].split()[0]},
                "awayTeam": {"name": r["visitor"].split()[-1], "abbreviation": None, "city": r["visitor"].split()[0]},
                "time": r.get("start_et"),
                "venue": None,
                "date": d.isoformat(),
                "prediction": {"winner":"TBD","confidence":0,"spread":None,"overUnder":None},
                "confidence":0,
                "gameStatus":"scheduled"
            })
        current_week[d.strftime("%A")] = games
    return {"currentWeek": current_week,
            "orderedDays":[(start+timedelta(days=i)).strftime("%A") for i in range(7)],
            "weekDates":[(start+timedelta(days=i)).isoformat() for i in range(7)],
            "metadata":{"lastUpdated":datetime.now(tz=LOCAL_TZ).isoformat(),
                        "weekStartDate":start.isoformat(),
                        "weekEndDate":(start+timedelta(days=6)).isoformat(),
                        "totalGames":sum(len(v) for v in current_week.values())}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", help="YYYY-MM-DD anchor date (default: today)")
    ap.add_argument("--season-year", type=int, help="NBA season ending year, e.g. 2026")
    ap.add_argument("--min-delay", type=int, default=DEFAULT_MIN_DELAY)
    ap.add_argument("--max-delay", type=int, default=DEFAULT_MAX_DELAY)
    ap.add_argument("--between-pages", type=int, default=DEFAULT_BETWEEN_PAGES)
    args = ap.parse_args()

    anchor = datetime.strptime(args.start_date,"%Y-%m-%d").date() if args.start_date else datetime.now(tz=LOCAL_TZ).date()
    season_year = args.season_year or (anchor.year if anchor.month<=4 else anchor.year+1)

    df = fetch_full_season_df(season_year, args.min_delay, args.max_delay, args.between_pages)
    window, start = window_next7(df, anchor)
    payload = to_ui_json(window, start)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH,"w",encoding="utf-8") as f: json.dump(payload,f,indent=2)
    print(f"Saved week {payload['metadata']['weekStartDate']} → {payload['metadata']['weekEndDate']} "
          f"({payload['metadata']['totalGames']} games) → {OUTPUT_PATH}")

if __name__=="__main__":
    main()
