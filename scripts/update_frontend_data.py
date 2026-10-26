"""
NBA Prediction Tracking System V5 - daily pipeline.

1. Pull one league-wide team game log (current + previous season).
2. Build each team's last-10-game rolling stats from cached box scores.
3. Predict today's games -> frontend/public/data/today.json
4. Grade past predictions -> frontend/public/data/lastweek.json, performance.json

Box scores never change once a game is final, so they are cached on disk in
data/boxscore_cache/. After the first run, a day only fetches the box scores of
the previous night's games instead of ~20 per team.
"""
import argparse
import json
import os
import time
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import boxscoreadvancedv3, boxscoretraditionalv3, leaguegamefinder, scoreboardv2
from nba_api.stats.static import teams
from requests.exceptions import ConnectionError, ReadTimeout, Timeout

# Suppress the specific sklearn warning about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Custom headers to reduce rate-limiting / blocking by stats.nba.com
CUSTOM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
API_TIMEOUT = 30
API_DELAY = 0.8  # seconds between stats.nba.com calls


def retry_api_call(func, max_retries=3, base_delay=2.0, label="API call"):
    """Retry an API call with exponential backoff on connection errors."""
    for attempt in range(max_retries + 1):
        try:
            result = func()
            time.sleep(API_DELAY)
            return result
        except (ConnectionError, ConnectionResetError, Timeout, ReadTimeout, OSError) as e:
            if attempt == max_retries:
                print(f"  [ERROR] {label} failed after {max_retries + 1} attempts: {e}")
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [RETRY] {label} attempt {attempt + 1} failed ({type(e).__name__}), retrying in {delay}s...")
            time.sleep(delay)


# --- Config ---
MODEL_PATH = "models/hist_gbm_v5/model_v5.pkl"
SCALER_PATH = "models/hist_gbm_v5/scaler_v5.pkl"
PREDICTORS_PATH = "models/hist_gbm_v5/predictors_v5.pkl"
FEATURE_IMPORTANCE_PATH = "models/hist_gbm_v5/feature_importance_v5.pkl"
PREDICTIONS_HISTORY_PATH = "data/predictions_history.json"
BOXSCORE_CACHE_DIR = "data/boxscore_cache"
TODAY_JSON_PATH = "frontend/public/data/today.json"
LASTWEEK_JSON_PATH = "frontend/public/data/lastweek.json"
PERFORMANCE_JSON_PATH = "frontend/public/data/performance.json"

ROLLING_WINDOW = 10
# Public track record counts picks from this date on (earlier picks used the old feature pipeline)
TRACKING_START = "2026-10-26"
MIN_GAMES_FOR_PREDICTION = 5
# Bump when process_game_stats changes so stale cache entries are recomputed
FEATURE_VERSION = 2
# SEASON_ID prefixes: 2 = regular season, 4 = playoffs, 5 = play-in (training data has no preseason)
COMPETITIVE_SEASON_PREFIXES = ("2", "4", "5")

# Stats shown in the UI "key factors" panel: (base stat, label, higher is better, format)
KEY_FACTOR_STATS = [
    ("ortg", "Offensive rating", True, "num"),
    ("drtg", "Defensive rating", False, "num"),
    ("total", "Points per game", True, "num"),
    ("ts%", "True shooting %", True, "frac"),
    ("efg%", "Effective FG %", True, "frac"),
    ("fg%", "Field goal %", True, "frac"),
    ("3p%_opp", "Opp. 3P% allowed", False, "frac"),
    ("trb%", "Rebound %", True, "pct"),
    ("ast", "Assists", True, "num"),
    ("tov%", "Turnover %", False, "pct"),
    ("pts_max", "Top scorer points", True, "num"),
    ("blk_opp", "Shots blocked against", False, "num"),
]


# =============================================================================
# BOX SCORE -> TEAM-GAME STATS
# Mirrors the basketball-reference columns the model was trained on
# (see data/nba_games_raw.csv). Formulas were checked against that file.
# =============================================================================

def _parse_minutes(value):
    """'34:12' or 'PT34M12.00S' -> 34.2 minutes."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    if s.startswith("PT"):
        s = s[2:].rstrip("S")
        mins, _, secs = s.partition("M")
        return float(mins or 0) + float(secs or 0) / 60
    if ":" in s:
        mins, secs = s.split(":", 1)
        return float(mins or 0) + float(secs or 0) / 60
    return float(s)


def _safe_div(a, b):
    return a / b if b else 0.0


def _team_totals(t_row, a_row):
    def g(row, col, scale=1.0):
        if row is None or col not in row.index or pd.isna(row[col]):
            return 0.0
        return float(row[col]) * scale

    return {
        "fg": g(t_row, "fieldGoalsMade"), "fga": g(t_row, "fieldGoalsAttempted"),
        "fg%": g(t_row, "fieldGoalsPercentage"),
        "3p": g(t_row, "threePointersMade"), "3pa": g(t_row, "threePointersAttempted"),
        "3p%": g(t_row, "threePointersPercentage"),
        "ft": g(t_row, "freeThrowsMade"), "fta": g(t_row, "freeThrowsAttempted"),
        "ft%": g(t_row, "freeThrowsPercentage"),
        "orb": g(t_row, "reboundsOffensive"), "drb": g(t_row, "reboundsDefensive"),
        "trb": g(t_row, "reboundsTotal"), "ast": g(t_row, "assists"), "stl": g(t_row, "steals"),
        "blk": g(t_row, "blocks"), "tov": g(t_row, "turnovers"), "pf": g(t_row, "foulsPersonal"),
        "pts": g(t_row, "points"), "+/-": g(t_row, "plusMinusPoints"),
        "ts%": g(a_row, "trueShootingPercentage"), "efg%": g(a_row, "effectiveFieldGoalPercentage"),
        "orb%": g(a_row, "offensiveReboundPercentage", 100), "drb%": g(a_row, "defensiveReboundPercentage", 100),
        "trb%": g(a_row, "reboundPercentage", 100), "ast%": g(a_row, "assistPercentage", 100),
        "ortg": g(a_row, "offensiveRating"), "drtg": g(a_row, "defensiveRating"),
        "poss": g(a_row, "possessions"),
        "mp": (_parse_minutes(t_row["minutes"]) if t_row is not None and "minutes" in t_row.index else 0.0) or 240.0,
    }


def _add_derived_team_stats(tm, opp):
    """Rate stats nba_api's advanced box score does not provide."""
    opp_poss = opp["poss"] or _safe_div(100 * opp["pts"], opp["ortg"])
    return {
        "3par": _safe_div(tm["3pa"], tm["fga"]),
        "ftr": _safe_div(tm["fta"], tm["fga"]),
        "tov%": 100 * _safe_div(tm["tov"], tm["fga"] + 0.44 * tm["fta"] + tm["tov"]),
        "stl%": 100 * _safe_div(tm["stl"], opp_poss),
        "blk%": 100 * _safe_div(tm["blk"], opp["fga"] - opp["3pa"]),
        "usg%": 100.0,
        "total": tm["pts"],
    }


def _player_drtg(p, tm, opp, team_poss):
    """Dean Oliver individual defensive rating (basketball-reference definition)."""
    mp = p["mp"]
    if mp <= 0 or not opp["fga"] or not team_poss:
        return None
    dor = _safe_div(opp["orb"], opp["orb"] + tm["drb"])
    dfg = _safe_div(opp["fg"], opp["fga"])
    fmwt = _safe_div(dfg * (1 - dor), dfg * (1 - dor) + (1 - dfg) * dor)
    opp_ft_miss = (1 - _safe_div(opp["ft"], opp["fta"])) if opp["fta"] else 0.0
    stops1 = p["stl"] + p["blk"] * fmwt * (1 - 1.07 * dor) + p["drb"] * (1 - fmwt)
    stops2 = (
        ((opp["fga"] - opp["fg"] - tm["blk"]) / tm["mp"]) * fmwt * (1 - 1.07 * dor)
        + ((opp["tov"] - tm["stl"]) / tm["mp"])
    ) * mp + _safe_div(p["pf"], tm["pf"]) * 0.4 * opp["fta"] * opp_ft_miss ** 2
    stop_pct = ((stops1 + stops2) * opp["mp"]) / (team_poss * mp)
    team_drtg = 100 * opp["pts"] / team_poss
    scoring_poss = opp["fg"] + (1 - opp_ft_miss ** 2) * opp["fta"] * 0.4
    d_pts_per_scposs = _safe_div(opp["pts"], scoring_poss)
    return team_drtg + 0.2 * (100 * d_pts_per_scposs * (1 - stop_pct) - team_drtg)


def _player_maxes(p_trad, p_adv, tm, opp):
    """Per-team max over players, matching basketball-reference box score maxes."""
    if p_trad.empty:
        return {}
    df = p_trad.copy()
    if not p_adv.empty:
        adv_cols = ["personId", "trueShootingPercentage", "effectiveFieldGoalPercentage",
                    "offensiveReboundPercentage", "defensiveReboundPercentage", "reboundPercentage",
                    "assistPercentage", "usagePercentage", "offensiveRating"]
        df = df.merge(p_adv[[c for c in adv_cols if c in p_adv.columns]], on="personId", how="left")

    col = lambda name: pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in df.columns else pd.Series(0.0, index=df.index)
    players = pd.DataFrame({
        "mp": df["minutes"].map(_parse_minutes) if "minutes" in df.columns else 0.0,
        "fg": col("fieldGoalsMade"), "fga": col("fieldGoalsAttempted"), "fg%": col("fieldGoalsPercentage"),
        "3p": col("threePointersMade"), "3pa": col("threePointersAttempted"), "3p%": col("threePointersPercentage"),
        "ft": col("freeThrowsMade"), "fta": col("freeThrowsAttempted"), "ft%": col("freeThrowsPercentage"),
        "orb": col("reboundsOffensive"), "drb": col("reboundsDefensive"), "trb": col("reboundsTotal"),
        "ast": col("assists"), "stl": col("steals"), "blk": col("blocks"), "tov": col("turnovers"),
        "pf": col("foulsPersonal"), "pts": col("points"), "+/-": col("plusMinusPoints"),
        "ts%": col("trueShootingPercentage"), "efg%": col("effectiveFieldGoalPercentage"),
        "orb%": col("offensiveReboundPercentage") * 100, "drb%": col("defensiveReboundPercentage") * 100,
        "trb%": col("reboundPercentage") * 100, "ast%": col("assistPercentage") * 100,
        "usg%": col("usagePercentage") * 100, "ortg": col("offensiveRating"),
    })
    played = players[players["mp"] > 0]
    if played.empty:
        return {}

    opp_poss = opp["poss"] or _safe_div(100 * opp["pts"], opp["ortg"])
    team_poss = np.mean([v for v in (tm["poss"], opp["poss"]) if v]) if (tm["poss"] or opp["poss"]) else 0.0
    shooters = played[played["fga"] > 0]
    tov_denom = played["fga"] + 0.44 * played["fta"] + played["tov"]
    derived = {
        "3par": (shooters["3pa"] / shooters["fga"]),
        "ftr": (shooters["fta"] / shooters["fga"]),
        "tov%": 100 * played["tov"][tov_denom > 0] / tov_denom[tov_denom > 0],
        "stl%": 100 * played["stl"] * (tm["mp"] / 5) / (played["mp"] * opp_poss) if opp_poss else pd.Series(dtype=float),
        "blk%": 100 * played["blk"] * (tm["mp"] / 5) / (played["mp"] * (opp["fga"] - opp["3pa"]))
                if opp["fga"] - opp["3pa"] else pd.Series(dtype=float),
        "drtg": pd.Series([_player_drtg(r, tm, opp, team_poss) for _, r in played.iterrows()], dtype=float),
    }

    maxes = {f"{k}_max": float(played[k].max()) for k in players.columns if k != "mp"}
    for k, series in derived.items():
        series = series.dropna()
        maxes[f"{k}_max"] = float(series.max()) if not series.empty else 0.0
    return maxes


def process_game_stats(trad, adv):
    """Return {team_id: stats} for both teams, each with the opponent's stats suffixed '_opp'."""
    t_rows = trad.team_stats.get_data_frame()
    p_rows = trad.player_stats.get_data_frame()
    a_t_rows = adv.team_stats.get_data_frame()
    a_p_rows = adv.player_stats.get_data_frame()

    team_ids = [int(t) for t in t_rows["teamId"].unique()]
    if len(team_ids) != 2:
        raise ValueError(f"expected 2 teams in box score, got {team_ids}")

    def first_row(df, team_id):
        rows = df[df["teamId"] == team_id]
        return rows.iloc[0] if not rows.empty else None

    totals = {tid: _team_totals(first_row(t_rows, tid), first_row(a_t_rows, tid)) for tid in team_ids}
    own = {}
    for tid in team_ids:
        oid = [t for t in team_ids if t != tid][0]
        stats = dict(totals[tid])
        stats.update(_add_derived_team_stats(totals[tid], totals[oid]))
        stats.update(_player_maxes(p_rows[p_rows["teamId"] == tid], a_p_rows[a_p_rows["teamId"] == tid],
                                   totals[tid], totals[oid]))
        for k in ("poss", "mp"):
            stats.pop(k, None)
        own[tid] = stats

    result = {}
    for tid in team_ids:
        oid = [t for t in team_ids if t != tid][0]
        merged = dict(own[tid])
        merged.update({f"{k}_opp": v for k, v in own[oid].items()})
        result[tid] = merged
    return result


def get_game_stats(game_id):
    """Box-score stats for both teams of a finished game, cached on disk."""
    path = os.path.join(BOXSCORE_CACHE_DIR, f"{game_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        if cached.get("v") == FEATURE_VERSION:
            return {int(k): v for k, v in cached["teams"].items()}

    trad = retry_api_call(
        lambda: boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, headers=CUSTOM_HEADERS, timeout=API_TIMEOUT),
        label=f"BoxScoreTrad(game={game_id})",
    )
    adv = retry_api_call(
        lambda: boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id, headers=CUSTOM_HEADERS, timeout=API_TIMEOUT),
        label=f"BoxScoreAdv(game={game_id})",
    )
    stats = process_game_stats(trad, adv)

    os.makedirs(BOXSCORE_CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"v": FEATURE_VERSION, "teams": {str(k): v for k, v in stats.items()}}, f)
    return stats


# =============================================================================
# GAME LOG, ROLLING FEATURES, PREDICTION
# =============================================================================

def season_label(dt):
    start = dt.year if dt.month >= 8 else dt.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def fetch_game_log(target_date):
    """All competitive team-games of the current and previous season played before target_date."""
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    frames = []
    for season in (season_label(target_dt.replace(year=target_dt.year - 1)), season_label(target_dt)):
        finder = retry_api_call(
            lambda s=season: leaguegamefinder.LeagueGameFinder(
                player_or_team_abbreviation="T", season_nullable=s, league_id_nullable="00",
                headers=CUSTOM_HEADERS, timeout=API_TIMEOUT,
            ),
            label=f"LeagueGameFinder(season={season})",
        )
        frames.append(finder.get_data_frames()[0])
    log = pd.concat(frames, ignore_index=True)
    log = log[log["SEASON_ID"].astype(str).str.startswith(COMPETITIVE_SEASON_PREFIXES)]
    log = log[log["GAME_DATE"] < target_date]
    log = log[log["WL"].notna()]  # finished games only
    return log.drop_duplicates(subset=["GAME_ID", "TEAM_ID"]).sort_values("GAME_DATE")


def rolling_stats(team_id, game_log):
    """Mean of the team's last 10 games' box-score stats, keys suffixed '_10'."""
    recent = game_log[game_log["TEAM_ID"] == team_id].tail(ROLLING_WINDOW)
    rows = []
    for game_id in recent["GAME_ID"]:
        try:
            rows.append(get_game_stats(game_id)[int(team_id)])
        except Exception as e:
            print(f"  [WARN] Skipping game {game_id}: {e}")
    if len(rows) < MIN_GAMES_FOR_PREDICTION:
        return None
    mean = pd.DataFrame(rows).mean()
    return {f"{k}_10": float(v) for k, v in mean.items()}


class Predictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.predictors = joblib.load(PREDICTORS_PATH)
        try:
            self.importances = joblib.load(FEATURE_IMPORTANCE_PATH)
        except FileNotFoundError:
            self.importances = {}
        self.feature_range = {
            f: (lo, hi) for f, lo, hi in zip(self.predictors, self.scaler.data_min_, self.scaler.data_max_)
        }

    def _row(self, team_roll, opp_roll, home_next):
        row = {f"{k}_team": v for k, v in team_roll.items()}
        row.update({f"{k}_opp": v for k, v in opp_roll.items()})
        row["home_next"] = home_next
        missing = [p for p in self.predictors if p not in row]
        if missing:
            raise ValueError(f"missing model features: {missing}")
        return np.array([[row[p] for p in self.predictors]])

    def home_win_prob(self, home_roll, away_roll):
        """The model was trained on both teams' perspectives; average them for one consistent probability."""
        p_home_view = self.model.predict_proba(self.scaler.transform(self._row(home_roll, away_roll, 1)))[0, 1]
        p_away_view = self.model.predict_proba(self.scaler.transform(self._row(away_roll, home_roll, 0)))[0, 1]
        return float((p_home_view + (1 - p_away_view)) / 2)

    def key_factors(self, home_roll, away_roll, limit=6):
        """Readable stat comparisons, ranked by model importance x size of the gap (in training-range units)."""
        factors = []
        for base, label, higher_better, fmt in KEY_FACTOR_STATS:
            key = f"{base}_10"
            if key not in home_roll or key not in away_roll:
                continue
            h, a = home_roll[key], away_roll[key]
            feats = [f for f in (f"{key}_team", f"{key}_opp") if f in self.feature_range]
            weight = 0.0
            if feats:
                lo, hi = self.feature_range[feats[0]]
                gap = abs(h - a) / (hi - lo) if hi > lo else 0.0
                weight = gap * sum(self.importances.get(f, 0.0) for f in feats)
            edge = "even" if np.isclose(h, a) else ("home" if (h > a) == higher_better else "away")
            factors.append({"label": label, "home": round(h, 3), "away": round(a, 3), "format": fmt,
                            "higherIsBetter": higher_better, "edge": edge, "weight": round(weight, 5)})
        factors.sort(key=lambda f: -f["weight"])
        return factors[:limit]


def team_info(team_id):
    t = teams.find_team_name_by_id(int(team_id))
    return {"name": t["nickname"], "city": t["city"], "fullName": t["full_name"],
            "abbreviation": t["abbreviation"], "team_id": int(team_id)}


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def load_predictions_history():
    if os.path.exists(PREDICTIONS_HISTORY_PATH):
        with open(PREDICTIONS_HISTORY_PATH) as f:
            return json.load(f)
    return {}


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def get_schedule(date_str):
    """Games on date_str from the scoreboard (the only source for not-yet-played games)."""
    board = retry_api_call(
        lambda: scoreboardv2.ScoreboardV2(game_date=date_str, headers=CUSTOM_HEADERS, timeout=API_TIMEOUT),
        label=f"ScoreboardV2(date={date_str})",
    )
    games_df = board.get_data_frames()[0].drop_duplicates(subset=["GAME_ID"])
    games = []
    for _, row in games_df.iterrows():
        venue = row.get("ARENA_NAME")
        games.append({
            "game_id": row["GAME_ID"], "home_id": int(row["HOME_TEAM_ID"]), "away_id": int(row["VISITOR_TEAM_ID"]),
            "status": row["GAME_STATUS_TEXT"].strip(), "venue": venue if isinstance(venue, str) else "",
        })
    return games


def predict_today(target_date, schedule, game_log, predictor, history):
    day_history = {g["game_id"]: g for g in history.get(target_date, [])}
    today_games = []
    print(f"\nGenerating predictions for {len(schedule)} games...")

    for game in schedule:
        home, away = team_info(game["home_id"]), team_info(game["away_id"])
        print(f"  {away['abbreviation']} @ {home['abbreviation']}")
        prediction = None

        existing = day_history.get(game["game_id"])
        if existing and existing.get("prediction"):
            # Never rewrite a prediction once published: keeps the tracked record honest
            prediction = existing["prediction"]
            print("    (already predicted)")
        else:
            try:
                home_roll = rolling_stats(game["home_id"], game_log)
                away_roll = rolling_stats(game["away_id"], game_log)
                if home_roll and away_roll:
                    p_home = predictor.home_win_prob(home_roll, away_roll)
                    home_pick = p_home >= 0.5
                    winner = home if home_pick else away
                    prediction = {
                        "winner": winner["fullName"],
                        "winnerAbbreviation": winner["abbreviation"],
                        "confidence": round(100 * max(p_home, 1 - p_home), 1),
                        "homeWinProbability": round(100 * p_home, 1),
                        "predictedHomeWin": bool(home_pick),
                        "keyFactors": predictor.key_factors(home_roll, away_roll),
                    }
                    print(f"    -> {winner['abbreviation']} ({prediction['confidence']}%)")
                else:
                    print("    [WARN] not enough recent games for a prediction")
            except Exception as e:
                print(f"    [ERROR] {e}")

        if prediction:
            day_history[game["game_id"]] = {
                "game_id": game["game_id"], "home_team_id": game["home_id"], "away_team_id": game["away_id"],
                "venue": game["venue"], "prediction": prediction,
                "actual_result": existing.get("actual_result") if existing else None,
            }
        today_games.append({
            "id": game["game_id"], "date": target_date, "time": game["status"], "venue": game["venue"],
            "gameStatus": "scheduled", "homeTeam": home, "awayTeam": away,
            "prediction": prediction,
        })

    history[target_date] = list(day_history.values())
    return today_games


def final_games_by_date(game_log):
    """{date: [{game_id, home_id, away_id, home_score, away_score}]} from the team game log."""
    out = {}
    for game_id, rows in game_log.groupby("GAME_ID"):
        home = rows[rows["MATCHUP"].str.contains(" vs. ", regex=False)]
        away = rows[rows["MATCHUP"].str.contains(" @ ", regex=False)]
        if home.empty or away.empty:
            continue
        h, a = home.iloc[0], away.iloc[0]
        out.setdefault(h["GAME_DATE"], []).append({
            "game_id": game_id, "home_id": int(h["TEAM_ID"]), "away_id": int(a["TEAM_ID"]),
            "home_score": int(h["PTS"]), "away_score": int(a["PTS"]),
        })
    return out


def grade_history(history, finals):
    """Attach actual results to every past prediction we can now grade."""
    graded = 0
    for date_str, games in history.items():
        results = {g["game_id"]: g for g in finals.get(date_str, [])}
        for hist_game in games:
            res = results.get(hist_game["game_id"])
            if not res or hist_game.get("actual_result"):
                continue
            winner_id = res["home_id"] if res["home_score"] > res["away_score"] else res["away_id"]
            w = team_info(winner_id)
            hist_game["actual_result"] = {
                "home_score": res["home_score"], "away_score": res["away_score"],
                "winner": w["fullName"],
            }
            graded += 1
    return graded


def build_last_week(target_date, finals, history):
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    week_dates = [(target_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7, 0, -1)]
    week = {"weekDates": week_dates, "orderedDays": [], "currentWeek": {}}

    for date_str in week_dates:
        day_name = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
        week["orderedDays"].append(day_name)
        preds = {g["game_id"]: g for g in history.get(date_str, [])}
        day_games = []
        for res in finals.get(date_str, []):
            home, away = team_info(res["home_id"]), team_info(res["away_id"])
            hist = preds.get(res["game_id"], {})
            game_obj = {
                "id": res["game_id"], "date": date_str, "time": "Final", "venue": hist.get("venue", ""),
                "gameStatus": "final", "homeTeam": home, "awayTeam": away,
                "finalScore": {"home": res["home_score"], "away": res["away_score"]},
                "prediction": None,
            }
            pred = hist.get("prediction")
            if pred:
                actual = hist.get("actual_result") or {}
                game_obj["prediction"] = {
                    "winner": pred["winner"], "confidence": pred["confidence"],
                    "homeWinProbability": pred.get("homeWinProbability"),
                    "predictedHomeWin": pred.get("predictedHomeWin"),
                    "correct": pred["winner"] == actual.get("winner") if actual else None,
                }
            day_games.append(game_obj)
        week["currentWeek"][day_name] = day_games
    return week


def build_performance(history, target_date):
    graded = []
    for date_str, games in history.items():
        if date_str < TRACKING_START:
            continue
        for g in games:
            actual = g.get("actual_result")
            if actual and actual.get("winner") and g.get("prediction"):
                pred = g["prediction"]
                home_won = actual["home_score"] > actual["away_score"]
                graded.append({"date": date_str, "correct": pred["winner"] == actual["winner"],
                               "confidence": float(pred["confidence"]), "home_won": home_won})

    def record(items):
        n, c = len(items), sum(i["correct"] for i in items)
        return {"games": n, "correct": c, "accuracy": round(c / n, 4) if n else None}

    cutoff = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    buckets = []
    for lo, hi in [(50, 60), (60, 70), (70, 80), (80, 101)]:
        items = [g for g in graded if lo <= g["confidence"] < hi]
        buckets.append({"range": f"{lo}–{min(hi, 100)}%", **record(items),
                        "avgConfidence": round(np.mean([i["confidence"] for i in items]), 1) if items else None})
    by_day = {}
    for g in graded:
        by_day.setdefault(g["date"], []).append(g)

    return {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "trackingSince": TRACKING_START,
        "overall": record(graded),
        "last7Days": record([g for g in graded if cutoff <= g["date"] < target_date]),
        "homeTeamBaseline": round(np.mean([g["home_won"] for g in graded]), 4) if graded else None,
        "byConfidence": buckets,
        "daily": [{"date": d, **record(items)} for d, items in sorted(by_day.items())][-30:],
    }


def main():
    parser = argparse.ArgumentParser(description="NBA Prediction Tracking System V5")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()

    target_date = datetime.now().strftime("%Y-%m-%d")
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD")
            return 1

    print("=" * 60)
    print(f"NBA PREDICTION TRACKING SYSTEM V5 - {target_date}")
    print("=" * 60)

    predictor = Predictor()
    history = load_predictions_history()
    print(f"  [OK] Model loaded, {len(history)} days of prediction history")

    # One call per season covers both the rolling-stat inputs and last week's results
    print("\n[FETCH] League game log...")
    try:
        game_log = fetch_game_log(target_date)
    except Exception as e:
        print(f"  [ERROR] Could not fetch the game log, leaving published data unchanged: {e}")
        return 1
    print(f"  {len(game_log)} team-games through {game_log['GAME_DATE'].max()}")

    finals = final_games_by_date(game_log)
    print(f"  Graded {grade_history(history, finals)} past predictions")

    failed = False
    print(f"\n[FETCH] Schedule for {target_date}...")
    try:
        schedule = get_schedule(target_date)
        print(f"  {len(schedule)} games")
        today_games = predict_today(target_date, schedule, game_log, predictor, history)
        write_json(TODAY_JSON_PATH, {
            "date": target_date, "generatedAt": datetime.now().isoformat(timespec="seconds"), "games": today_games,
        })
        print(f"\n[OK] Saved {TODAY_JSON_PATH}")
    except Exception as e:
        print(f"  [ERROR] Could not build today's predictions: {e}")
        failed = True

    write_json(LASTWEEK_JSON_PATH, build_last_week(target_date, finals, history))
    print(f"[OK] Saved {LASTWEEK_JSON_PATH}")
    write_json(PERFORMANCE_JSON_PATH, build_performance(history, target_date))
    print(f"[OK] Saved {PERFORMANCE_JSON_PATH}")
    write_json(PREDICTIONS_HISTORY_PATH, history)
    print("[OK] Updated predictions history")

    print("\n" + "=" * 60)
    print("COMPLETE (with errors)" if failed else "COMPLETE!")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
