import pandas as pd
import numpy as np
import joblib
import time
import warnings
import json
import os
import argparse
import random
from functools import wraps
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3, boxscoreadvancedv3, scoreboardv2
from nba_api.stats.static import teams
import requests

# Suppress the specific sklearn warning about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# =============================================================================
# NBA API CONFIGURATION - Fix for GitHub Actions
# =============================================================================
# The NBA API blocks requests from cloud providers (GitHub Actions, AWS, etc.)
# Per nba_api v1.1.0+ documentation, we must pass headers and timeout to each endpoint call

# Browser-like headers to bypass NBA.com cloud IP blocking
NBA_API_HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}

# Timeout in seconds for all NBA API requests
NBA_API_TIMEOUT = 90

# Proxy configuration for GitHub Actions (bypasses cloud provider IP blocking)
# Set PROXY_URL environment variable (format: http://username:password@host:port)
NBA_API_PROXY = os.getenv('PROXY_URL')
if NBA_API_PROXY:
    NBA_API_PROXIES = {
        'http': NBA_API_PROXY,
        'https': NBA_API_PROXY
    }
    print(f"[Proxy] Configured: {NBA_API_PROXY.split('@')[1] if '@' in NBA_API_PROXY else NBA_API_PROXY}")
else:
    NBA_API_PROXIES = None
    print("[Proxy] Not configured (running locally)")

# =============================================================================
# RETRY LOGIC FOR NBA API
# =============================================================================

def retry_on_timeout(max_retries=5, initial_delay=2):
    """Decorator to retry functions on timeout/connection errors with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    # Add random jitter to avoid thundering herd
                    if attempt > 0:
                        jitter = random.uniform(0, delay * 0.3)
                        sleep_time = delay + jitter
                        print(f"    Retry {attempt}/{max_retries} after {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                    
                    result = func(*args, **kwargs)
                    
                    # Add delay between successful calls to avoid rate limiting
                    time.sleep(random.uniform(0.6, 1.2))
                    return result
                    
                except (requests.exceptions.ReadTimeout, 
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout) as e:
                    last_exception = e
                    print(f"    ⚠️ Timeout/Connection error on attempt {attempt + 1}: {str(e)[:100]}")
                    delay *= 2  # Exponential backoff
                    
                    if attempt == max_retries - 1:
                        print(f"    ❌ Failed after {max_retries} attempts")
                        raise
                        
                except Exception as e:
                    # Don't retry on other exceptions
                    print(f"    ❌ Non-retryable error: {str(e)[:100]}")
                    raise
            
            raise last_exception
        return wrapper
    return decorator

# --- Config ---
MODEL_PATH = "models/hist_gbm_v5/model_v5.pkl"
SCALER_PATH = "models/hist_gbm_v5/scaler_v5.pkl"
PREDICTORS_PATH = "models/hist_gbm_v5/predictors_v5.pkl"
PREDICTIONS_HISTORY_PATH = "data/predictions_history.json"
TODAY_JSON_PATH = "frontend/public/data/today.json"
LASTWEEK_JSON_PATH = "frontend/public/data/lastweek.json"

# =============================================================================
# HELPER FUNCTIONS (from predict_v5.py)
# =============================================================================

@retry_on_timeout(max_retries=5, initial_delay=2)
def fetch_rolling_stats(team_id):
    """Fetch rolling stats for a team (last 10 games)"""
    gamefinder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        headers=NBA_API_HEADERS,
        timeout=NBA_API_TIMEOUT,
        proxy=NBA_API_PROXIES
    )
    games = gamefinder.get_data_frames()[0]
    
    # Sort and take last 10 completed games
    games = games.sort_values("GAME_DATE").tail(15) 
    last_10 = games.tail(10)
    stats_list = []
    
    for i, row in last_10.iterrows():
        g_id = row["GAME_ID"]
        try:
            trad = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=g_id,
                headers=NBA_API_HEADERS,
                timeout=NBA_API_TIMEOUT,
                proxy=NBA_API_PROXIES
            )
            adv = boxscoreadvancedv3.BoxScoreAdvancedV3(
                game_id=g_id,
                headers=NBA_API_HEADERS,
                timeout=NBA_API_TIMEOUT,
                proxy=NBA_API_PROXIES
            )
            time.sleep(0.4) 
            game_stats = process_game_stats(trad, adv, team_id)
            stats_list.append(game_stats)
        except Exception as e:
            continue
            
    if not stats_list:
        return None
        
    df = pd.DataFrame(stats_list)
    rolling = df.mean().to_frame().T
    cols = {c: f"{c}_10" for c in df.columns}
    rolling = rolling.rename(columns=cols)
    
    return rolling

def process_game_stats(trad, adv, team_id):
    """Process game stats from traditional and advanced box scores"""
    t_rows = trad.team_stats.get_data_frame()
    p_rows = trad.player_stats.get_data_frame()
    a_t_rows = adv.team_stats.get_data_frame()
    a_p_rows = adv.player_stats.get_data_frame()
    
    all_teams = t_rows["teamId"].unique()
    opp_id = [t for t in all_teams if t != team_id][0] if len(all_teams) > 1 else None
    
    team_row_t = t_rows[t_rows["teamId"] == team_id]
    team_row_a = a_t_rows[a_t_rows["teamId"] == team_id]
    
    # Simple Helper
    def val(df, col):
        if not df.empty and col in df.columns:
            return float(df.iloc[0][col])
        return 0.0

    stats = {}
    # Traditional
    stats["fg"] = val(team_row_t, "fieldGoalsMade")
    stats["fga"] = val(team_row_t, "fieldGoalsAttempted")
    stats["fg%"] = val(team_row_t, "fieldGoalsPercentage")
    stats["3p"] = val(team_row_t, "threePointersMade")
    stats["3pa"] = val(team_row_t, "threePointersAttempted")
    stats["3p%"] = val(team_row_t, "threePointersPercentage")
    stats["ft"] = val(team_row_t, "freeThrowsMade")
    stats["fta"] = val(team_row_t, "freeThrowsAttempted")
    stats["ft%"] = val(team_row_t, "freeThrowsPercentage")
    stats["orb"] = val(team_row_t, "reboundsOffensive")
    stats["drb"] = val(team_row_t, "reboundsDefensive")
    stats["trb"] = val(team_row_t, "reboundsTotal")
    stats["ast"] = val(team_row_t, "assists")
    stats["stl"] = val(team_row_t, "steals")
    stats["blk"] = val(team_row_t, "blocks")
    stats["tov"] = val(team_row_t, "turnovers")
    stats["pf"] = val(team_row_t, "foulsPersonal")
    stats["pts"] = val(team_row_t, "points")
    stats["+/-"] = val(team_row_t, "plusMinusPoints")
    
    # Advanced
    stats["ts%"] = val(team_row_a, "trueShootingPercentage")
    stats["efg%"] = val(team_row_a, "effectiveFieldGoalPercentage")
    stats["3par"] = val(team_row_a, "threePointersAttemptedPercentage")
    stats["ftr"] = val(team_row_a, "freeThrowsAttemptedRate")
    stats["orb%"] = val(team_row_a, "offensiveReboundPercentage") * 100
    stats["drb%"] = val(team_row_a, "defensiveReboundPercentage") * 100
    stats["trb%"] = val(team_row_a, "reboundPercentage") * 100
    stats["ast%"] = val(team_row_a, "assistPercentage") * 100
    stats["stl%"] = val(team_row_a, "stealPercentage") * 100
    stats["blk%"] = val(team_row_a, "blockPercentage") * 100
    stats["tov%"] = val(team_row_a, "turnoverPercentage") * 100
    stats["usg%"] = 100.0
    stats["ortg"] = val(team_row_a, "offensiveRating")
    stats["drtg"] = val(team_row_a, "defensiveRating")
    
    # Opponent
    if opp_id:
        opp_row_t = t_rows[t_rows["teamId"] == opp_id]
        opp_row_a = a_t_rows[a_t_rows["teamId"] == opp_id]
        
        stats["mp_opp"] = 240.0
        stats["fg_opp"] = val(opp_row_t, "fieldGoalsMade")
        stats["fga_opp"] = val(opp_row_t, "fieldGoalsAttempted")
        stats["fg%_opp"] = val(opp_row_t, "fieldGoalsPercentage")
        stats["3p_opp"] = val(opp_row_t, "threePointersMade")
        stats["3pa_opp"] = val(opp_row_t, "threePointersAttempted")
        stats["3p%_opp"] = val(opp_row_t, "threePointersPercentage")
        stats["ft_opp"] = val(opp_row_t, "freeThrowsMade")
        stats["fta_opp"] = val(opp_row_t, "freeThrowsAttempted")
        stats["ft%_opp"] = val(opp_row_t, "freeThrowsPercentage")
        stats["orb_opp"] = val(opp_row_t, "reboundsOffensive")
        stats["drb_opp"] = val(opp_row_t, "reboundsDefensive")
        stats["trb_opp"] = val(opp_row_t, "reboundsTotal")
        stats["ast_opp"] = val(opp_row_t, "assists")
        stats["stl_opp"] = val(opp_row_t, "steals")
        stats["blk_opp"] = val(opp_row_t, "blocks")
        stats["tov_opp"] = val(opp_row_t, "turnovers")
        stats["pf_opp"] = val(opp_row_t, "foulsPersonal")
        stats["pts_opp"] = val(opp_row_t, "points")
        stats["+/-_opp"] = val(opp_row_t, "plusMinusPoints")
        
        stats["ts%_opp"] = val(opp_row_a, "trueShootingPercentage")
        stats["efg%_opp"] = val(opp_row_a, "effectiveFieldGoalPercentage")
        stats["3par_opp"] = val(opp_row_a, "threePointersAttemptedPercentage")
        stats["ftr_opp"] = val(opp_row_a, "freeThrowsAttemptedRate")
        stats["orb%_opp"] = val(opp_row_a, "offensiveReboundPercentage") * 100
        stats["drb%_opp"] = val(opp_row_a, "defensiveReboundPercentage") * 100
        stats["trb%_opp"] = val(opp_row_a, "reboundPercentage") * 100
        stats["ast%_opp"] = val(opp_row_a, "assistPercentage") * 100
        stats["stl%_opp"] = val(opp_row_a, "stealPercentage") * 100
        stats["blk%_opp"] = val(opp_row_a, "blockPercentage") * 100
        stats["tov%_opp"] = val(opp_row_a, "turnoverPercentage") * 100
        stats["usg%_opp"] = 100.0
        stats["ortg_opp"] = val(opp_row_a, "offensiveRating")
        stats["drtg_opp"] = val(opp_row_a, "defensiveRating")
        
        stats["mp_opp.1_opp"] = 240.0

    # Max Stats
    my_p_rows_t = p_rows[p_rows["teamId"] == team_id]
    my_p_rows_a = a_p_rows[a_p_rows["teamId"] == team_id]
    
    def get_max(df, api_key):
        if not df.empty and api_key in df.columns:
            return float(df[api_key].max())
        return 0.0

    trad_map = {"fg": "fieldGoalsMade", "fga": "fieldGoalsAttempted", "fg%": "fieldGoalsPercentage", "3p": "threePointersMade", "3pa": "threePointersAttempted", "3p%": "threePointersPercentage", "ft": "freeThrowsMade", "fta": "freeThrowsAttempted", "ft%": "freeThrowsPercentage", "orb": "reboundsOffensive", "drb": "reboundsDefensive", "trb": "reboundsTotal", "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers", "pf": "foulsPersonal", "pts": "points", "+/-": "plusMinusPoints"}
    adv_map = {"ts%": "trueShootingPercentage", "efg%": "effectiveFieldGoalPercentage", "3par": "threePointersAttemptedPercentage", "ftr": "freeThrowsAttemptedRate", "orb%": ("offensiveReboundPercentage", 100), "drb%": ("defensiveReboundPercentage", 100), "trb%": ("reboundPercentage", 100), "ast%": ("assistPercentage", 100), "stl%": ("stealPercentage", 100), "blk%": ("blockPercentage", 100), "tov%": ("turnoverPercentage", 100), "usg%": ("usagePercentage", 100), "ortg": ("offensiveRating", 1), "drtg": ("defensiveRating", 1)}

    for k, api in trad_map.items():
        stats[f"{k}_max"] = get_max(my_p_rows_t, api)
    for k, val in adv_map.items():
        if isinstance(val, tuple):
            stats[f"{k}_max"] = get_max(my_p_rows_a, val[0]) * val[1]
        else:
            stats[f"{k}_max"] = get_max(my_p_rows_a, val)

    if opp_id:
        opp_p_rows_t = p_rows[p_rows["teamId"] == opp_id]
        opp_p_rows_a = a_p_rows[a_p_rows["teamId"] == opp_id]
        for k, api in trad_map.items():
            stats[f"{k}_max_opp"] = get_max(opp_p_rows_t, api)
        for k, val in adv_map.items():
            if isinstance(val, tuple):
                stats[f"{k}_max_opp"] = get_max(opp_p_rows_a, val[0]) * val[1]
            else:
                stats[f"{k}_max_opp"] = get_max(opp_p_rows_a, val)

    return stats

def predict_matchup(home_id, away_id, model, scaler, predictors):
    """Generate prediction for a matchup"""
    try:
        home_name = teams.find_team_name_by_id(home_id)['full_name']
        away_name = teams.find_team_name_by_id(away_id)['full_name']
    except:
        home_name = f"ID {home_id}"
        away_name = f"ID {away_id}"
        
    print(f"  Analyzing: {home_name} vs {away_name}...")
    
    home_rolling = fetch_rolling_stats(home_id)
    away_rolling = fetch_rolling_stats(away_id)
    
    if home_rolling is None or away_rolling is None:
        return None
        
    home_cols = {c: f"{c}_team" for c in home_rolling.columns}
    home_stats = home_rolling.rename(columns=home_cols)
    away_cols = {c: f"{c}_opp" for c in away_rolling.columns}
    away_stats = away_rolling.rename(columns=away_cols)
    
    input_row = pd.concat([home_stats.reset_index(drop=True), away_stats.reset_index(drop=True)], axis=1)
    input_row["home_next"] = 1
    
    for p in predictors:
        if p not in input_row.columns:
            input_row[p] = 0.0
            
    X = input_row[predictors]
    X_scaled = scaler.transform(X.values)
    
    score = model.decision_function(X_scaled)[0]
    if score > 0:
        winner = home_name
        prob = 1 / (1 + np.exp(-score))
        predicted_home_win = True
    else:
        winner = away_name
        prob = 1 / (1 + np.exp(score))
        predicted_home_win = False
        
    confidence = prob * 100
    
    print(f"  → {winner} ({confidence:.1f}%)")
    
    return {
        "winner": winner,
        "confidence": round(confidence, 1),
        "predictedHomeWin": predicted_home_win
    }

# =============================================================================
# MAIN WORKFLOW FUNCTIONS
# =============================================================================

def load_predictions_history():
    """Load or create predictions history file"""
    if os.path.exists(PREDICTIONS_HISTORY_PATH):
        with open(PREDICTIONS_HISTORY_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_predictions_history(history):
    """Save predictions history to file"""
    os.makedirs(os.path.dirname(PREDICTIONS_HISTORY_PATH), exist_ok=True)
    with open(PREDICTIONS_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

@retry_on_timeout(max_retries=5, initial_delay=3)
def get_games_for_date(date_str):
    """Fetch games for a specific date"""
    print(f"\n📅 Fetching games for {date_str}...")
    
    board = scoreboardv2.ScoreboardV2(
        game_date=date_str,
        headers=NBA_API_HEADERS,
        timeout=NBA_API_TIMEOUT,
        proxy=NBA_API_PROXIES
    )
    games_df = board.get_data_frames()[0]  # GameHeader dataframe
    linescore_df = board.get_data_frames()[1]  # LineScore dataframe (has PTS!)
    
    if games_df.empty:
        print("  No games found for this date.")
        return []
    
    game_list = []
    for i, row in games_df.iterrows():
        game_id = row["GAME_ID"]
        home_id = row["HOME_TEAM_ID"]
        away_id = row["VISITOR_TEAM_ID"]
        status = row["GAME_STATUS_TEXT"]
        venue = row.get("ARENA_NAME", "") if pd.notna(row.get("ARENA_NAME")) else ""
        
        # Get scores from LineScore dataframe
        game_scores = linescore_df[linescore_df["GAME_ID"] == game_id]
        
        home_score = None
        away_score = None
        
        if not game_scores.empty:
            home_row = game_scores[game_scores["TEAM_ID"] == home_id]
            away_row = game_scores[game_scores["TEAM_ID"] == away_id]
            
            if not home_row.empty:
                pts = home_row.iloc[0]["PTS"]
                home_score = int(pts) if pd.notna(pts) and pts > 0 else None
            
            if not away_row.empty:
                pts = away_row.iloc[0]["PTS"]
                away_score = int(pts) if pd.notna(pts) and pts > 0 else None
        
        game_list.append({
            "game_id": game_id,
            "home_id": home_id,
            "away_id": away_id,
            "status": status,
            "venue": venue,
            "home_score": home_score,
            "away_score": away_score
        })
        
    print(f"  Found {len(game_list)} games.")
    return game_list

def generate_todays_predictions(target_date, model, scaler, predictors, history):
    """Generate predictions for target date's games and update history"""
    games = get_games_for_date(target_date)
    
    if not games:
        return None
    
    print(f"\n🔮 Generating predictions for {len(games)} games...")
    
    today_games = []
    predictions_for_history = []
    
    for game in games:
        try:
            home_team = teams.find_team_name_by_id(game["home_id"])
            away_team = teams.find_team_name_by_id(game["away_id"])
            
            pred_result = predict_matchup(game["home_id"], game["away_id"], model, scaler, predictors)
            
            game_obj = {
                "id": game["game_id"],
                "date": target_date,
                "time": game["status"],
                "venue": game.get("venue", ""),
                "gameStatus": "scheduled",
                "homeTeam": {
                    "name": home_team['nickname'],
                    "city": home_team['city'],
                    "abbreviation": home_team['abbreviation'],
                    "team_id": game["home_id"]
                },
                "awayTeam": {
                    "name": away_team['nickname'],
                    "city": away_team['city'],
                    "abbreviation": away_team['abbreviation'],
                    "team_id": game["away_id"]
                },
                "prediction": pred_result if pred_result else {"winner": "TBD", "confidence": 0, "predictedHomeWin": None}
            }
            
            today_games.append(game_obj)
            
            # Save to history
            if pred_result:
                predictions_for_history.append({
                    "game_id": game["game_id"],
                    "home_team_id": game["home_id"],
                    "away_team_id": game["away_id"],
                    "prediction": pred_result,
                    "actual_result": None
                })
                
        except Exception as e:
            print(f"  ⚠️ Error processing game {game['game_id']}: {e}")
            continue
    
    # Update history
    history[target_date] = predictions_for_history
    
    # Create today.json
    today_json = {
        "date": target_date,
        "games": today_games
    }
    
    return today_json

def fetch_last_7_days_results(target_date, history):
    """Fetch results for last 7 days from target_date and evaluate predictions"""
    print("\n📊 Fetching last 7 days of results...")
    
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    # Reverse to go from oldest to newest (7 days ago -> yesterday)
    last_7_days = [(target_dt - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7, 0, -1)]
    
    week_data = {
        "weekDates": last_7_days,
        "orderedDays": [],
        "currentWeek": {}
    }
    
    for date_str in last_7_days:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day_name = date_obj.strftime('%A')
        week_data["orderedDays"].append(day_name)
        
        print(f"\n  {day_name} ({date_str}):")
        
        # Fetch games for that date
        try:
            games = get_games_for_date(date_str)
            
            if not games:
                print("    No games on this day")
                week_data["currentWeek"][day_name] = []
                continue
            
            day_games = []
            
            for game in games:
                game_id = game["game_id"]
                home_id = game["home_id"]
                away_id = game["away_id"]
                home_score = game["home_score"]
                away_score = game["away_score"]
                
                home_team = teams.find_team_name_by_id(home_id)
                away_team = teams.find_team_name_by_id(away_id)
                
                # Determine actual winner
                if home_score is not None and away_score is not None:
                    actual_winner = home_team['full_name'] if home_score > away_score else away_team['full_name']
                else:
                    actual_winner = None
                
                # Look up prediction from history
                prediction = None
                correct = None
                
                if date_str in history:
                    for hist_game in history[date_str]:
                        if hist_game["game_id"] == game_id:
                            prediction = hist_game["prediction"]
                            
                            # Evaluate
                            if actual_winner and prediction:
                                correct = (prediction["winner"] == actual_winner)
                                
                                # Update history with actual result
                                hist_game["actual_result"] = {
                                    "home_score": home_score,
                                    "away_score": away_score,
                                    "winner": actual_winner
                                }
                            break
                
                game_obj = {
                    "id": game_id,
                    "date": date_str,
                    "time": game["status"],
                    "venue": game.get("venue", ""),
                    "gameStatus": "final",
                    "homeTeam": {
                        "name": home_team['nickname'],
                        "city": home_team['city'],
                        "abbreviation": home_team['abbreviation'],
                        "team_id": home_id
                    },
                    "awayTeam": {
                        "name": away_team['nickname'],
                        "city": away_team['city'],
                        "abbreviation": away_team['abbreviation'],
                        "team_id": away_id
                    },
                    "finalScore": {
                        "home": home_score,
                        "away": away_score
                    }
                }
                
                if prediction:
                    game_obj["prediction"] = {
                        "winner": prediction["winner"],
                        "confidence": prediction["confidence"],
                        "correct": correct
                    }
                
                day_games.append(game_obj)
                
                # Print evaluation
                if correct is not None:
                    status = "✅" if correct else "❌"
                    print(f"    {status} {home_team['abbreviation']} {home_score}-{away_score} {away_team['abbreviation']}: Predicted {prediction['winner']}")
                elif home_score and away_score:
                    print(f"    ℹ️  {home_team['abbreviation']} {home_score}-{away_score} {away_team['abbreviation']}: No prediction found")
            
            week_data["currentWeek"][day_name] = day_games
            
        except Exception as e:
            print(f"    ⚠️ Error fetching games for {date_str}: {e}")
            week_data["currentWeek"][day_name] = []
            continue
    
    return week_data

def main():
    parser = argparse.ArgumentParser(description='NBA Prediction Tracking System V5')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD). Defaults to today.')
    args = parser.parse_args()
    
    # Determine target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').strftime('%Y-%m-%d')
            print(f"🎯 Running for custom date: {target_date}")
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            return
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("🏀 NBA PREDICTION TRACKING SYSTEM V5")
    print("=" * 60)
    print(f"\n💡 Using browser headers on all NBA API requests (timeout: {NBA_API_TIMEOUT}s)")
    
    # Load model
    print("\n⚙️  Loading model...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        predictors = joblib.load(PREDICTORS_PATH)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return
    
    # Load predictions history
    history = load_predictions_history()
    print(f"  ✓ Loaded predictions history ({len(history)} days tracked)")
    
    # Generate today's predictions
    today_json = generate_todays_predictions(target_date, model, scaler, predictors, history)
    
    if today_json:
        os.makedirs(os.path.dirname(TODAY_JSON_PATH), exist_ok=True)
        with open(TODAY_JSON_PATH, 'w') as f:
            json.dump(today_json, f, indent=2)
        print(f"\n✓ Saved today's predictions to {TODAY_JSON_PATH}")
    
    # Fetch and evaluate last 7 days
    lastweek_json = fetch_last_7_days_results(target_date, history)
    
    os.makedirs(os.path.dirname(LASTWEEK_JSON_PATH), exist_ok=True)
    with open(LASTWEEK_JSON_PATH, 'w') as f:
        json.dump(lastweek_json, f, indent=2)
    print(f"\n✓ Saved last week's results to {LASTWEEK_JSON_PATH}")
    
    # Save updated history
    save_predictions_history(history)
    print(f"✓ Updated predictions history")
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
