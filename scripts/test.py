import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3, boxscoreadvancedv3
import joblib
import os

# Configuration
TEAM_ID = 1610612737  # Atlanta Hawks
TEAM_ABBREVIATION = "ATL" # Used to identify team in boxscore
HOME_NEXT_GAME = 1 # 1 if next game is Home, 0 if Away. (Hardcoded for now, or fetch schedule)

def get_recent_games(team_id, limit=15):
    print(f"Fetching recent games for Team ID {team_id}...")
    finder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, player_or_team_abbreviation="T")
    df = finder.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False)
    return df.head(limit)

def get_game_stats(game_id, team_id):
    time.sleep(0.6) # Rate limit
    try:
        box_trad = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        team_trad = box_trad.team_stats.get_data_frame()
        player_trad = box_trad.player_stats.get_data_frame()
        
        time.sleep(0.6) # Rate limit
        box_adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
        team_adv = box_adv.team_stats.get_data_frame()
        player_adv = box_adv.player_stats.get_data_frame()

    except Exception as e:
        print(f"Error fetching game {game_id}: {e}")
        return None

    if team_trad.empty or team_adv.empty:
        return None

    # Identify Team and Opponent
    # Traditional
    team_row_t = team_trad[team_trad["teamId"] == team_id]
    opp_row_t = team_trad[team_trad["teamId"] != team_id]
    
    # Advanced
    team_row_a = team_adv[team_adv["teamId"] == team_id]
    opp_row_a = team_adv[team_adv["teamId"] != team_id]
    
    if team_row_t.empty or team_row_a.empty: 
        return None
    
    # Identify Players
    team_players_t = player_trad[player_trad["teamId"] == team_id]
    opp_players_t = player_trad[player_trad["teamId"] != team_id]
    
    team_players_a = player_adv[player_adv["teamId"] == team_id]
    opp_players_a = player_adv[player_adv["teamId"] != team_id]

    # Mapping NBA API columns to our Model columns (lowercase, simplified)
    # Model columns: fg, fga, fg%, 3p, 3pa, 3p%, ft, fta, ft%, orb, drb, trb, ast, stl, blk, tov, pf, pts
    # Plus _max versions.
    # Plus _opp versions.
    
    def extract_stats(t_row_t, p_rows_t, t_row_a, p_rows_a, suffix=""):
        stats = {}
        
        # Basic Team Stats (from Team Row)
        # API keys: fieldGoalsMade, fieldGoalsAttempted, fieldGoalsPercentage, etc.
        # Check API keys first... they are usually CamelCase in v3
        
        # Helper to safely get value
        def get_val(row, key):
            if key in row.columns and not row.empty:
                return row[key].iloc[0]
            return 0

        # Team Stats
        stats[f"fg{suffix}"] = get_val(t_row_t, "fieldGoalsMade")
        stats[f"fga{suffix}"] = get_val(t_row_t, "fieldGoalsAttempted")
        stats[f"fg%{suffix}"] = get_val(t_row_t, "fieldGoalsPercentage")
        
        stats[f"3p{suffix}"] = get_val(t_row_t, "threePointersMade")
        stats[f"3pa{suffix}"] = get_val(t_row_t, "threePointersAttempted")
        stats[f"3p%{suffix}"] = get_val(t_row_t, "threePointersPercentage")
        
        stats[f"ft{suffix}"] = get_val(t_row_t, "freeThrowsMade")
        stats[f"fta{suffix}"] = get_val(t_row_t, "freeThrowsAttempted")
        stats[f"ft%{suffix}"] = get_val(t_row_t, "freeThrowsPercentage")
        
        stats[f"orb{suffix}"] = get_val(t_row_t, "reboundsOffensive")
        stats[f"drb{suffix}"] = get_val(t_row_t, "reboundsDefensive")
        stats[f"trb{suffix}"] = get_val(t_row_t, "reboundsTotal")
        
        stats[f"ast{suffix}"] = get_val(t_row_t, "assists")
        stats[f"stl{suffix}"] = get_val(t_row_t, "steals")
        stats[f"blk{suffix}"] = get_val(t_row_t, "blocks")
        stats[f"tov{suffix}"] = get_val(t_row_t, "turnovers")
        stats[f"pf{suffix}"] = get_val(t_row_t, "foulsPersonal")
        stats[f"pts{suffix}"] = get_val(t_row_t, "points")
        stats[f"+/-{suffix}"] = get_val(t_row_t, "plusMinusPoints")
        
        # Advanced Stats
        # Important: CSV has `orb%`, `usg%`, etc. as 0-100. API gives 0.0-1.0.
        # But `ts%`, `efg%` in CSV are 0.0-1.0. 
        # Check carefully!
        
        # 0-100 Scale stats
        stats[f"orb%{suffix}"] = get_val(t_row_a, "offensiveReboundPercentage") * 100
        stats[f"drb%{suffix}"] = get_val(t_row_a, "defensiveReboundPercentage") * 100
        stats[f"trb%{suffix}"] = get_val(t_row_a, "reboundPercentage") * 100
        stats[f"ast%{suffix}"] = get_val(t_row_a, "assistPercentage") * 100
        stats[f"stl%{suffix}"] = get_val(t_row_a, "stealPercentage") * 100
        stats[f"blk%{suffix}"] = get_val(t_row_a, "blockPercentage") * 100
        stats[f"tov%{suffix}"] = get_val(t_row_a, "turnoverPercentage") * 100
        stats[f"usg%{suffix}"] = 100.0 # Team usage is always 100%
        
        # 0-1 Scale stats (Decimals)
        stats[f"ts%{suffix}"] = get_val(t_row_a, "trueShootingPercentage")
        stats[f"efg%{suffix}"] = get_val(t_row_a, "effectiveFieldGoalPercentage")
        stats[f"3par{suffix}"] = get_val(t_row_a, "threePointersAttemptedPercentage") 
        stats[f"ftr{suffix}"] = get_val(t_row_a, "freeThrowsAttemptedRate")
        
        # Ratings (Values ~100)
        stats[f"ortg{suffix}"] = get_val(t_row_a, "offensiveRating")
        stats[f"drtg{suffix}"] = get_val(t_row_a, "defensiveRating")
        
        # Advanced/Calculated Stats if model needs them (ts%, etc.)
        # For now, we stick to basics + max. 
        
        # Player Max Stats
        # Merging Traditional and Advanced players
        
        # Traditional Keys
        trad_keys = {
            "fg": "fieldGoalsMade", "fga": "fieldGoalsAttempted", "fg%": "fieldGoalsPercentage",
            "3p": "threePointersMade", "3pa": "threePointersAttempted", "3p%": "threePointersPercentage",
            "ft": "freeThrowsMade", "fta": "freeThrowsAttempted", "ft%": "freeThrowsPercentage",
            "orb": "reboundsOffensive", "drb": "reboundsDefensive", "trb": "reboundsTotal",
            "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers",
            "pf": "foulsPersonal", "pts": "points", "+/-": "plusMinusPoints"
        }
        
        # Advanced Keys (0-100 Scale)
        adv_keys_100 = {
            "orb%": "offensiveReboundPercentage", "drb%": "defensiveReboundPercentage",
            "trb%": "reboundPercentage", "ast%": "assistPercentage", "stl%": "stealPercentage",
            "blk%": "blockPercentage", "tov%": "turnoverPercentage", "usg%": "usagePercentage"
        }
        
        # Advanced Keys (0-1 Scale or Value)
        adv_keys_val = {
            "ts%": "trueShootingPercentage", "efg%": "effectiveFieldGoalPercentage",
            "ortg": "offensiveRating", "drtg": "defensiveRating"
        }

        # Calculate Traditional Max
        if not p_rows_t.empty:
            for my_key, api_key in trad_keys.items():
                if api_key in p_rows_t.columns:
                    stats[f"{my_key}_max{suffix}"] = p_rows_t[api_key].max()
                else:
                    stats[f"{my_key}_max{suffix}"] = 0
        else:
             for my_key in trad_keys:
                stats[f"{my_key}_max{suffix}"] = 0
                
        # Calculate Advanced Max
        if not p_rows_a.empty:
            # 0-100 group
            for my_key, api_key in adv_keys_100.items():
                if api_key in p_rows_a.columns:
                    stats[f"{my_key}_max{suffix}"] = p_rows_a[api_key].max() * 100
                else:
                    stats[f"{my_key}_max{suffix}"] = 0
            
            # Value group
            for my_key, api_key in adv_keys_val.items():
                if api_key in p_rows_a.columns:
                    stats[f"{my_key}_max{suffix}"] = p_rows_a[api_key].max()
                else:
                    stats[f"{my_key}_max{suffix}"] = 0
        else:
            for my_key in list(adv_keys_100.keys()) + list(adv_keys_val.keys()):
                stats[f"{my_key}_max{suffix}"] = 0

        return stats

    # Extract for Team
    row_stats = extract_stats(team_row_t, team_players_t, team_row_a, team_players_a, suffix="")
    # Extract for Opponent
    opp_stats = extract_stats(opp_row_t, opp_players_t, opp_row_a, opp_players_a, suffix="_opp")
    
    row_stats.update(opp_stats)
    return row_stats

def main():
    # 1. Load Artifacts
    print("Loading model artifacts...")
    try:
        model = joblib.load("models/ridge_regression/model_optimized.pkl")
        scaler = joblib.load("models/ridge_regression/scaler_optimized.pkl")
        predictors = joblib.load("models/ridge_regression/predictors_optimized.pkl")
    except FileNotFoundError:
        print("Model artifacts not found. Please run scripts/optimize_model_v2.py first.")
        return

    # 2. Get Data
    recent_games = get_recent_games(TEAM_ID)
    
    # We need last 10 games to compute ONE prediction row?
    # Yes, we need the rolling average of the *last 10 games*.
    # So we fetch the actual last 10 finished games.
    
    games_to_process = recent_games.head(10)
    
    print(f"Processing last {len(games_to_process)} games to compute rolling stats...")
    
    aggregated_stats = []
    
    for _, row in games_to_process.iterrows():
        game_id = row["GAME_ID"]
        game_date = row["GAME_DATE"]
        print(f"Fetching stats for game {game_id} ({game_date})...")
        
        stats = get_game_stats(game_id, TEAM_ID)
        if stats:
            aggregated_stats.append(stats)
            
    if not aggregated_stats:
        print("No stats collected.")
        return

    # 3. Compute Rolling Average
    df_games = pd.DataFrame(aggregated_stats)
    
    # We want the ONE row representing the "Average of last 10 games"
    # This row will be the input to our model to predict the NEXT game.
    
    avg_stats = df_games.mean().to_dict() # Series to dict
    
    # 4. Prepare Feature Vector
    # The model expects columns named like "fg_10", "fg_opp_10", etc.
    # Our extract_stats gave "fg", "fg_opp".
    # We need to rename them to match the rolling naming convention used in optimization.
    
    input_row = {}
    for k, v in avg_stats.items():
        input_row[f"{k}_10"] = v
        
    # Add scalar features
    input_row["home_next"] = HOME_NEXT_GAME
    
    # Create DataFrame
    input_df = pd.DataFrame([input_row])
    
    # Ensure all predictors are present
    # Some might be missing (e.g., if we didn't calculate 'usg%_10').
    # We fill missing with 0.
    
    for p in predictors:
        if p not in input_df.columns:
            # print(f"Warning: Missing predictor {p}, filling with 0")
            input_df[p] = 0
            
    # Select only predictors (order matters? Sklearn usually robust if names match, but safer to order)
    # Actually, Scaler expects ALL columns it was trained on if it was trained on the full set.
    # In V2, we trained scaler on `predictors_candidates`.
    # `predictors_candidates` = list(data_rolling.columns) + ["home_next"]
    # So we need to provide ALL rolling columns + home_next.
    
    # We assume 'predictors_candidates' are ALL columns we generated.
    # Wait, 'scaler' in v2 was fitted on `final_data[predictors_candidates]`.
    # SFS selected `predictors` FROM that.
    # So to use the model, we first Scale (using all candidates), then filter (using predictors).
    # We need to know `predictors_candidates` (the feature names the scaler expects).
    # Since we can't easily load that list unless we saved it, we might have an issue if the scaler complains about feature count.
    # Standard scaler expects the same number of features.
    
    # Hack: Inspect scaler.n_features_in_
    # But names?
    # Ideally, we should have saved the VALID COLUMNS list or relevant list.
    # But generally, if we generate *all* stats we can, and fill others, validation might pass.
    # However, `MinMaxScaler` just works on array shape. It doesn't check column names.
    # This is dangerous.
    
    # BETTER APPROACH for test.py:
    # We really should have trained the scaler on JUST the chosen predictors in V2.
    # (I mentioned this in thought process but didn't implement it in V2 because filtering happens after SFS).
    
    # FIX: I will update V2 to save a meaningful scaler (re-fit on predictors).
    # Or, `test.py` just runs raw? No, Ridge needs scaling.
    
    # I'll rely on `test.py` generating a dataframe, and hopefully matching enough.
    # Actually, without the exact column order the scaler expects, `scaler.transform` on a DataFrame is unreliable if columns aren't sorted same way.
    # `scaler.feature_names_in_` (if sklearn > 1.0) stores names.
    
    try:
        # Re-order input_df to match scaler's expected features
        if hasattr(scaler, "feature_names_in_"):
            expected_cols = scaler.feature_names_in_
            # Add missing
            for c in expected_cols:
                if c not in input_df.columns:
                    input_df[c] = 0
            # Reorder
            input_df = input_df[expected_cols]
            
            # Cast to float to avoid type issues
            input_df = input_df.astype(float)
            
            # Scale (pass values to avoid feature name strictness issues)
            scaled_features = scaler.transform(input_df.values)
            
            # Convert back to DF to select predictors by name
            # SFS used column names? 
            # In V2: `predictors = list(np.array(predictors_candidates)[sfs.get_support()])`
            # `model.fit(train[predictors], ...)`
            # So the model expects `predictors` columns.
            
            # If we scaled `input_df` (which has `expected_cols`), we get an array.
            # We need to pick the columns corresponding to `predictors` from this scaled array.
            
            scaled_df = pd.DataFrame(scaled_features, columns=expected_cols)
            final_input = scaled_df[predictors]
            
            # Predict
            probs = model.decision_function(final_input) # RidgeClassifier doesn't have predict_proba by default?
            pred = model.predict(final_input)[0]
            
            print("\n=== PREDICTION ===")
            print(f"Prediction: {'WIN' if pred == 1 else 'LOSS'}")
            print(f"Model Confidence/Score: {probs[0]:.4f}")
            print("==================")
            
            # Print significant stats
            print("\nKey Predictor Values (Scaled):")
            for p in predictors[:5]: # Show top 5
                print(f"{p}: {final_input[p].iloc[0]:.4f}")

    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()