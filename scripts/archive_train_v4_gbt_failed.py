
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuration ---
DATA_PATH = "data/nba_games_raw.csv"
MODEL_DIR = "models/gradient_boosting_v4"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("date")
    df = df.reset_index(drop=True)
    return df

def clean_data(df):
    print("Cleaning data...")
    # Remove low value columns
    del df["mp"]
    del df["mp.1"]
    if "item" in df.columns:
        del df["item"]
    
    # Drop columns with nulls
    nulls = pd.isnull(df).sum()
    nulls = nulls[nulls > 0]
    valid_cols = df.columns[~df.columns.isin(nulls.index)]
    df = df[valid_cols].copy()
    
    return df

def add_target(team_df):
    team_df["target"] = team_df["won"].shift(-1)
    return team_df

def compute_rolling_averages(df):
    print("Computing Rolling Averages...")
    # Numeric columns to roll
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_remove = ["season", "target", "won", "team", "team_opp", "date", "home_next"] 
    
    if "home" in numeric_cols: numeric_cols.remove("home")
    
    for c in cols_to_remove:
        if c in numeric_cols:
            numeric_cols.remove(c)
            
    def find_team_averages(team):
        rolling = team[numeric_cols].rolling(10).mean()
        return rolling

    df_rolling = df.groupby(["team"], group_keys=False).apply(find_team_averages)
    
    # Rename rolling cols
    rolling_cols = [f"{c}_10" for c in df_rolling.columns]
    df_rolling.columns = rolling_cols
    
    # Concatenate back
    df_final = pd.concat([df, df_rolling], axis=1)
    df_final = df_final.dropna()
    
    return df_final, rolling_cols

def merge_opponent_stats(df):
    # Not used directly, integrated into main logic similar to V3
    pass

def main():
    df = load_data()
    df = clean_data(df)
    
    # Add target
    df = df.groupby("team", group_keys=False).apply(add_target)
    # Ensure target is int
    df["target"] = df["target"].fillna(0).astype(int)
    
    # Add scalar features
    df["home_next"] = df["home"].shift(-1)
    df["team_opp_next"] = df["team_opp"].shift(-1)
    df["date_next"] = df["date"].shift(-1)
    
    df = df.dropna()
    
    df, rolling_cols = compute_rolling_averages(df)
    
    # Merge Opponent Data (Matchup Merge)
    stats_cols = rolling_cols + ["team", "date_next"]
    right_df = df[stats_cols].copy()
    
    # Merge rule: Team A (Left) vs Team B (Right)
    # Left: [team_opp_next, date_next] matches Right: [team, date_next]
    
    combined = pd.merge(
        df, 
        right_df, 
        left_on=["team_opp_next", "date_next"], 
        right_on=["team", "date_next"],
        suffixes=("_team", "_opp")
    )
    
    # Feature Selection Preparation
    potential_predictors = [c for c in combined.columns if "_10" in c or c == "home_next"]
    
    # Clean NaNs
    combined = combined.dropna()
    
    print(f"Training Data Shape: {combined.shape}")
    print(f"Num Predictors: {len(potential_predictors)}")
    
    # Validation checks
    if len(potential_predictors) == 0:
        print("ERROR: No predictors found!")
        return

    if combined[potential_predictors].isnull().values.any():
        print("ERROR: NaNs in predictors!")
        return
        
    non_numeric = combined[potential_predictors].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"ERROR: Non-numeric predictors: {non_numeric}")
        return

    # Scale fit on ALL potential predictors first (for SFS)
    print("Scaling...")
    combined_raw = combined.copy()
    
    scaler_full = MinMaxScaler()
    combined[potential_predictors] = scaler_full.fit_transform(combined[potential_predictors])
    
    # SFS with Gradient Boosting is too slow (O(N^2)). 
    # Switching to SelectFromModel (Feature Importance) which is standard for Trees.
    
    print("Fitting initial GBT for Feature Selection...")
    gb_sel = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_sel.fit(combined[potential_predictors], combined["target"])
    
    # Select top 30 based on importance
    # We can use SelectFromModel with max_features (sklearn > 1.1) or threshold
    # Let's perform manual selection to guarantee exactly 30
    
    importances = gb_sel.feature_importances_
    indices = np.argsort(importances)[::-1] # Descending
    top_30_idx = indices[:30]
    
    selected_predictors = [potential_predictors[i] for i in top_30_idx]
    
    # selector = SelectFromModel(gb_sel, max_features=30, threshold=-np.inf, prefit=True)
    # selected_mask = selector.get_support()
    # selected_predictors = list(np.array(potential_predictors)[selected_mask])
    
    print(f"Selected Predictors (Top 30 by Importance): {selected_predictors}")
    
    # Retrain Scaler and Model on Selected ONLY
    print("Retraining Final Model & Scaler on Selected Features...")
    final_scaler = MinMaxScaler()
    X_selected = final_scaler.fit_transform(combined_raw[selected_predictors])
    
    # Fit Final Model
    gb_final = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_final.fit(X_selected, combined["target"])
    
    # Save
    print("Saving Artifacts...")
    joblib.dump(gb_final, f"{MODEL_DIR}/model_v4.pkl")
    joblib.dump(final_scaler, f"{MODEL_DIR}/scaler_v4.pkl")
    joblib.dump(selected_predictors, f"{MODEL_DIR}/predictors_v4.pkl")
    
    # Check Accuracy
    preds = gb_final.predict(X_selected)
    score = accuracy_score(combined["target"], preds)
    print(f"Training Accuracy: {score:.4f}")

if __name__ == "__main__":
    main()
