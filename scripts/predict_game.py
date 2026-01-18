import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3, boxscoreadvancedv3
import joblib

# Configuration
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hist_gbm_v5"
MODEL_PATH = MODEL_DIR / "model_v5.pkl"
SCALER_PATH = MODEL_DIR / "scaler_v5.pkl"
PREDICTORS_PATH = MODEL_DIR / "predictors_v5.pkl"

# Load model artifacts once when module is imported
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    predictors = joblib.load(PREDICTORS_PATH)
    print(f"✓ Loaded model artifacts from {MODEL_DIR}")
except FileNotFoundError as e:
    print(f"✗ Model artifacts not found: {e}")
    model, scaler, predictors = None, None, None


def get_recent_games(team_id, limit=10):
    """Fetch recent games for a team."""
    try:
        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id, 
            player_or_team_abbreviation="T"
        )
        df = finder.get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE", ascending=False)
        return df.head(limit)
    except Exception as e:
        print(f"Error fetching recent games for team {team_id}: {e}")
        return pd.DataFrame()


def get_game_stats(game_id, team_id):
    """Fetch detailed stats for a specific game."""
    time.sleep(0.6)  # Rate limit
    try:
        # Fetch traditional box score
        box_trad = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        team_trad = box_trad.team_stats.get_data_frame()
        player_trad = box_trad.player_stats.get_data_frame()
        
        time.sleep(0.6)  # Rate limit
        # Fetch advanced box score
        box_adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
        team_adv = box_adv.team_stats.get_data_frame()
        player_adv = box_adv.player_stats.get_data_frame()

    except Exception as e:
        print(f"Error fetching game {game_id}: {e}")
        return None

    if team_trad.empty or team_adv.empty:
        return None

    # Identify Team and Opponent
    team_row_t = team_trad[team_trad["teamId"] == team_id]
    opp_row_t = team_trad[team_trad["teamId"] != team_id]
    team_row_a = team_adv[team_adv["teamId"] == team_id]
    opp_row_a = team_adv[team_adv["teamId"] != team_id]
    
    if team_row_t.empty or team_row_a.empty:
        return None
    
    # Extract stats
    team_players_t = player_trad[player_trad["teamId"] == team_id]
    opp_players_t = player_trad[player_trad["teamId"] != team_id]
    team_players_a = player_adv[player_adv["teamId"] == team_id]
    opp_players_a = player_adv[player_adv["teamId"] != team_id]

    def extract_stats(t_row_t, p_rows_t, t_row_a, p_rows_a, suffix=""):
        stats = {}
        
        def get_val(row, key):
            if key in row.columns and not row.empty:
                return row[key].iloc[0]
            return 0

        # Basic team stats
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
        
        # Advanced stats (convert to 0-100 scale)
        stats[f"orb%{suffix}"] = get_val(t_row_a, "offensiveReboundPercentage") * 100
        stats[f"drb%{suffix}"] = get_val(t_row_a, "defensiveReboundPercentage") * 100
        stats[f"trb%{suffix}"] = get_val(t_row_a, "reboundPercentage") * 100
        stats[f"ast%{suffix}"] = get_val(t_row_a, "assistPercentage") * 100
        stats[f"stl%{suffix}"] = get_val(t_row_a, "stealPercentage") * 100
        stats[f"blk%{suffix}"] = get_val(t_row_a, "blockPercentage") * 100
        stats[f"tov%{suffix}"] = get_val(t_row_a, "turnoverPercentage") * 100
        stats[f"usg%{suffix}"] = 100.0  # Team usage is always 100%
        
        # Efficiency stats (0-1 scale)
        stats[f"ts%{suffix}"] = get_val(t_row_a, "trueShootingPercentage")
        stats[f"efg%{suffix}"] = get_val(t_row_a, "effectiveFieldGoalPercentage")
        stats[f"3par{suffix}"] = get_val(t_row_a, "threePointersAttemptedPercentage")
        stats[f"ftr{suffix}"] = get_val(t_row_a, "freeThrowsAttemptedRate")
        
        # Ratings
        stats[f"ortg{suffix}"] = get_val(t_row_a, "offensiveRating")
        stats[f"drtg{suffix}"] = get_val(t_row_a, "defensiveRating")
        
        # Player max stats
        trad_keys = {
            "fg": "fieldGoalsMade", "fga": "fieldGoalsAttempted", "fg%": "fieldGoalsPercentage",
            "3p": "threePointersMade", "3pa": "threePointersAttempted", "3p%": "threePointersPercentage",
            "ft": "freeThrowsMade", "fta": "freeThrowsAttempted", "ft%": "freeThrowsPercentage",
            "orb": "reboundsOffensive", "drb": "reboundsDefensive", "trb": "reboundsTotal",
            "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers",
            "pf": "foulsPersonal", "pts": "points", "+/-": "plusMinusPoints"
        }
        
        adv_keys_100 = {
            "orb%": "offensiveReboundPercentage", "drb%": "defensiveReboundPercentage",
            "trb%": "reboundPercentage", "ast%": "assistPercentage", "stl%": "stealPercentage",
            "blk%": "blockPercentage", "tov%": "turnoverPercentage", "usg%": "usagePercentage"
        }
        
        adv_keys_val = {
            "ts%": "trueShootingPercentage", "efg%": "effectiveFieldGoalPercentage",
            "ortg": "offensiveRating", "drtg": "defensiveRating"
        }

        # Traditional max
        if not p_rows_t.empty:
            for my_key, api_key in trad_keys.items():
                if api_key in p_rows_t.columns:
                    stats[f"{my_key}_max{suffix}"] = p_rows_t[api_key].max()
                else:
                    stats[f"{my_key}_max{suffix}"] = 0
        else:
            for my_key in trad_keys:
                stats[f"{my_key}_max{suffix}"] = 0
                
        # Advanced max
        if not p_rows_a.empty:
            for my_key, api_key in adv_keys_100.items():
                if api_key in p_rows_a.columns:
                    stats[f"{my_key}_max{suffix}"] = p_rows_a[api_key].max() * 100
                else:
                    stats[f"{my_key}_max{suffix}"] = 0
            
            for my_key, api_key in adv_keys_val.items():
                if api_key in p_rows_a.columns:
                    stats[f"{my_key}_max{suffix}"] = p_rows_a[api_key].max()
                else:
                    stats[f"{my_key}_max{suffix}"] = 0
        else:
            for my_key in list(adv_keys_100.keys()) + list(adv_keys_val.keys()):
                stats[f"{my_key}_max{suffix}"] = 0

        return stats

    # Extract for team and opponent
    row_stats = extract_stats(team_row_t, team_players_t, team_row_a, team_players_a, suffix="")
    opp_stats = extract_stats(opp_row_t, opp_players_t, opp_row_a, opp_players_a, suffix="_opp")
    
    row_stats.update(opp_stats)
    return row_stats


def compute_team_rolling_stats(team_id, num_games=10):
    """Compute rolling average stats for a team based on their last N games."""
    recent_games = get_recent_games(team_id, limit=num_games)
    
    if recent_games.empty:
        print(f"No recent games found for team {team_id}")
        return None
    
    aggregated_stats = []
    
    for _, row in recent_games.iterrows():
        game_id = row["GAME_ID"]
        stats = get_game_stats(game_id, team_id)
        if stats:
            aggregated_stats.append(stats)
            
    if not aggregated_stats:
        print(f"No stats collected for team {team_id}")
        return None

    # Compute rolling average
    df_games = pd.DataFrame(aggregated_stats)
    avg_stats = df_games.mean().to_dict()
    
    return avg_stats


def predict_game(home_team_id, away_team_id, home_team_name, away_team_name):
    """
    Predict the outcome of a game between two teams.
    
    Returns:
        dict: {
            "winner": str (team name),
            "confidence": float (0-100),
            "predictedHomeWin": bool
        }
        or None if prediction fails
    """
    if model is None or scaler is None or predictors is None:
        print("Model not loaded, cannot make predictions")
        return None
    
    print(f"\nGenerating prediction for {away_team_name} @ {home_team_name}...")
    
    # Get rolling stats for both teams
    print(f"  Fetching stats for {home_team_name}...")
    home_stats = compute_team_rolling_stats(home_team_id, num_games=10)
    
    if home_stats is None:
        print(f"  Failed to get stats for {home_team_name}")
        return None
    
    print(f"  Fetching stats for {away_team_name}...")
    away_stats = compute_team_rolling_stats(away_team_id, num_games=10)
    
    if away_stats is None:
        print(f"  Failed to get stats for {away_team_name}")
        return None
    
    # Prepare feature vector
    # Home team stats with "_10" suffix, opponent stats become away team stats
    input_row = {}
    
    for k, v in home_stats.items():
        input_row[f"{k}_10"] = v
    
    # Add home indicator
    input_row["home_next"] = 1
    
    # Create DataFrame
    input_df = pd.DataFrame([input_row])
    
    # Ensure all predictors are present
    for p in predictors:
        if p not in input_df.columns:
            input_df[p] = 0
    
    try:
        # Re-order to match scaler's expected features
        if hasattr(scaler, "feature_names_in_"):
            expected_cols = scaler.feature_names_in_
            for c in expected_cols:
                if c not in input_df.columns:
                    input_df[c] = 0
            input_df = input_df[expected_cols]
            
            input_df = input_df.astype(float)
            scaled_features = scaler.transform(input_df.values)
            scaled_df = pd.DataFrame(scaled_features, columns=expected_cols)
            final_input = scaled_df[predictors]
        else:
            # Fallback if scaler doesn't have feature names
            final_input = input_df[predictors]
        
        # Make prediction
        prediction = model.predict(final_input)[0]
        
        # Get confidence (probability)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(final_input)[0]
            confidence = max(proba) * 100  # Convert to percentage
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(final_input)[0]
            # Convert decision function to pseudo-probability
            confidence = min(max(50 + decision * 10, 50), 95)
        else:
            confidence = 60.0  # Default confidence
        
        # Determine winner
        predicted_home_win = bool(prediction == 1)
        winner = home_team_name if predicted_home_win else away_team_name
        
        print(f"  ✓ Prediction: {winner} wins ({confidence:.1f}% confidence)")
        
        return {
            "winner": winner,
            "confidence": round(confidence, 1),
            "predictedHomeWin": predicted_home_win
        }
        
    except Exception as e:
        print(f"  ✗ Prediction failed: {e}")
        return None
