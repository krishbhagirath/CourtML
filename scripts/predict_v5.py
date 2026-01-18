import pandas as pd
import numpy as np
import joblib
import time
import warnings
import json
import os
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3, boxscoreadvancedv3, scoreboardv2
from nba_api.stats.static import teams

# Suppress the specific sklearn warning about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Config ---
MODEL_PATH = "models/hist_gbm_v5/model_v5.pkl"
SCALER_PATH = "models/hist_gbm_v5/scaler_v5.pkl"
PREDICTORS_PATH = "models/hist_gbm_v5/predictors_v5.pkl"

def get_todays_games():
    # Fetch today's scoreboard
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching schedule for {today}...")
    
    board = scoreboardv2.ScoreboardV2(game_date=today)
    games = board.get_data_frames()[0]
    
    if games.empty:
        print("No games found for today!")
        return []
    
    game_list = []
    for i, row in games.iterrows():
        game_list.append({
            "game_id": row["GAME_ID"],
            "home_id": row["HOME_TEAM_ID"],
            "away_id": row["VISITOR_TEAM_ID"],
            "status": row["GAME_STATUS_TEXT"] # e.g. "7:00 pm ET"
        })
        
    return game_list

def fetch_rolling_stats(team_id):
    # print(f"Fetching stats for Team {team_id}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    games = gamefinder.get_data_frames()[0]
    
    # Sort and take last 10 completed games
    games = games.sort_values("GAME_DATE").tail(15) 
    
    last_10 = games.tail(10)
    stats_list = []
    
    for i, row in last_10.iterrows():
        g_id = row["GAME_ID"]
        try:
            trad = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=g_id)
            adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=g_id)
            time.sleep(0.4) 
            game_stats = process_game_stats(trad, adv, team_id)
            stats_list.append(game_stats)
        except Exception as e:
            # print(f"Error fetching {g_id}: {e}")
            continue
            
    if not stats_list:
        return None
        
    df = pd.DataFrame(stats_list)
    rolling = df.mean().to_frame().T
    cols = {c: f"{c}_10" for c in df.columns}
    rolling = rolling.rename(columns=cols)
    
    return rolling

def process_game_stats(trad, adv, team_id):
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

    # Max Stats (Simplified for brevity, assuming existing logic)
    # Re-implementing logic from prior steps for correctness
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
    try:
        home_name = teams.find_team_name_by_id(home_id)['full_name']
        away_name = teams.find_team_name_by_id(away_id)['full_name']
    except:
        home_name = f"ID {home_id}"
        away_name = f"ID {away_id}"
        
    print(f"\nAnalyzing: {home_name} vs {away_name}...")
    
    home_rolling = fetch_rolling_stats(home_id)
    away_rolling = fetch_rolling_stats(away_id)
    
    if home_rolling is None or away_rolling is None:
        return
        
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
    else:
        winner = away_name
        prob = 1 / (1 + np.exp(score))
        
    confidence = prob * 100
    
    print(f"PREDICTED: {winner} ({confidence:.1f}%)")
    
    return {
        "winner": winner,
        "confidence": round(confidence, 1),
        "updated_at": datetime.now().isoformat() 
    }

def main():
    print("=== NBA Match Predictor V5 (Dynamic) === ")
    
    games = get_todays_games()
    if not games:
        print("No games found.")
        return
    
    print("Loading V5 Model...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        predictors = joblib.load(PREDICTORS_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    predictions_map = {}
    print(f"Generating predictions for {len(games)} games...")
    
    for game in games:
        try:
            home_abbr = teams.find_team_name_by_id(game["home_id"])['abbreviation']
            away_abbr = teams.find_team_name_by_id(game["away_id"])['abbreviation']
            
            pred_result = predict_matchup(game["home_id"], game["away_id"], model, scaler, predictors)
            if pred_result:
                key = f"{home_abbr}_{away_abbr}"
                predictions_map[key] = pred_result
        except Exception as e:
            print(f"Skipping game {game['game_id']}: {e}")
            continue

    # Save Predictions
    output_path = "frontend/public/data/predictions.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions_map, f, indent=2)
    print(f"\nPredictions saved to {output_path}")

if __name__ == "__main__":
    main()
