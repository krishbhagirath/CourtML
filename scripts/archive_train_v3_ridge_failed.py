
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuration ---
DATA_PATH = "data/nba_games_raw.csv"
MODEL_DIR = "models/ridge_regression_v3"
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
    cols_to_remove = ["season", "target", "won", "team", "team_opp", "date", "home_next"] # exclude these from rolling
    
    # "home" is actually useful to roll? No, we want next game's home.
    # Actually, we want rolling stats of performance.
    
    # Keep `home` out of rolling, we use `home_next`
    if "home" in numeric_cols: numeric_cols.remove("home")
    
    for c in cols_to_remove:
        if c in numeric_cols:
            numeric_cols.remove(c)
            
    def find_team_averages(team):
        # Shift to avoid lookahead leakage
        rolling = team[numeric_cols].rolling(10).mean()
        return rolling

    df_rolling = df.groupby(["team"], group_keys=False).apply(find_team_averages)
    
    # Rename rolling cols
    rolling_cols = [f"{c}_10" for c in df_rolling.columns]
    df_rolling.columns = rolling_cols
    
    # Concatenate back
    # We need 'team', 'date' to merge later
    df_final = pd.concat([df, df_rolling], axis=1)
    
    # Drop rows where rolling is NaN (first 10 games)
    df_final = df_final.dropna()
    
    return df_final, rolling_cols

def merge_opponent_stats(df):
    print("Merging Opponent Stats...")
    # We want to merge the dataframe with ITSELF based on the Matchup.
    # Each game has two rows: Team A vs Team B, and Team B vs Team A.
    # They share the same `date` (and `game_id` if we had it, but we use `team_opp` and `date`).
    
    # Key: Merge on ["date", "team_opp"] == ["date", "team"]
    # So if Row 1 is ATL vs BOS on 2022-01-01.
    # We want to grab the row where team="BOS" and date="2022-01-01".
    
    # Helper for merging
    df_opp = df.copy()
    
    # Renaming certain columns so they don't collide or to make clear they are opponent stats
    # We primarily want the ROLLING columns of the opponent.
    # Getting 'fg_10' of the opponent => 'fg_10_opp'
    
    # We also want scalar features of the opponent? 
    # V2 only used 'home_next'.
    
    # Let's rename ALL columns in df_opp to _opp, except the merge keys.
    merge_keys = ["date", "team_opp"] # We map df["team_opp"] -> df_opp["team"]
    
    # We are merging on:
    # Left: [date, team_opp] (matches) Right: [date, team]
    
    # So we rename right side `team` to `team_opp` to match left side.
    
    right_df = df_opp.rename(columns={"team": "team_opp"})
    
    # Now merge.
    # Note: `team_opp` in Right DF was originally `team`.
    # `team_opp` in Left DF is the opponent name.
    # So merging on ["date", "team_opp"] joins "ATL's opponent BOS" with "BOS's row".
    
    final_merged = pd.merge(df, right_df, on=["date", "team_opp"], suffixes=("_team", "_opp"))
    
    return final_merged

def main():
    df = load_data()
    df = clean_data(df)
    
    # Add target
    df = df.groupby("team", group_keys=False).apply(add_target)
    df["target"] = df["target"].fillna(0).astype(int)
    
    # Add "home_next" column
    # We need to shift target?
    # Original notebook: `df["target"] = df["won"].shift(-1)`
    # This means rows contain stats up to Game G, and target is result of Game G+1.
    
    # But wait, original notebook logic:
    # df["target"][g] is outcome of g+1.
    # features[g] are stats of g.
    # No, usually we want input = Rolling(Game G..G-9), Target = Result(Game G+1).
    
    # Let's stick to the V2 logic which worked for setup:
    # 1. Rolling averages computed on raw history.
    # 2. Add Target (Next Game Result).
    # 3. Drop NaNs.
    
    # Add "home_next" column
    df["home_next"] = df["home"].shift(-1)
    df["team_opp_next"] = df["team_opp"].shift(-1)
    df["date_next"] = df["date"].shift(-1)
    
    df = df.dropna() # Remove rows where next game info is missing (prop last game)
    
    df, rolling_cols = compute_rolling_averages(df)
    
    # Now Merge Opponent Data
    # IMPORTANT: We need to merge based on the NEXT game.
    # Row i: Team A, Rolling Stats(0-10), Target=Win, Next Opponent = B, Next Date = D.
    # We need B's Rolling Stats at Date D (or D-1?).
    
    # Actually, the easier way that preserves the "Merge" structure:
    # Each row in `df` represents a game that HAS happened (or rather, the pre-game state).
    # IF we shifted target, we are predicting the next game.
    
    # Let's re-verify the "Matchup Merge" logic.
    # Ideally:
    # Dataframe has 1 row per game per team.
    # Row: Team A, Game G, Stats A.
    # Merge with: Team B, Game G, Stats B.
    # Result: Team A vs B, Stats A, Stats B.
    # Then we use this merged row to predict "Did A win?".
    
    # But we are doing "Rolling".
    # So for Game G+1 (Target):
    # Inputs: Rolling(A) entering G+1, Rolling(B) entering G+1.
    
    # So we need to match:
    # Left: Row i (Team A). `date_next`, `team_opp_next`.
    # Right: Row j (Team B). `date_next`, `team`. (Team B is the opponent).
    
    # Wait, `right_df` (Team B) should be indexed by when B plays A.
    # If we merge on `date_next`, we find the row for B where `date_next` is the same.
    # And `team` (of B) == `team_opp_next` (of A).
    
    # Let's prepare Right DF
    # We only need the rolling cols and identity for the Right DF
    stats_cols = rolling_cols + ["team", "date_next"]
    right_df = df[stats_cols].copy()
    
    combined = pd.merge(
        df, 
        right_df, 
        left_on=["team_opp_next", "date_next"], 
        right_on=["team", "date_next"],
        suffixes=("_team", "_opp")
    )
    
    # Now we have `fg_10_team` and `fg_10_opp`.
    
    # Feature Selection
    remove_from_predictors = ["season", "date", "won", "target", "team", "team_opp", "home", "team_opp_next", "date_next", "team_team", "team_opp", "home_team"]
    
    # We want valid predictors.
    # Basically all `_10_team` and `_10_opp` columns, plus `home_next`.
    
    potential_predictors = [c for c in combined.columns if "_10" in c or c == "home_next"]
    
    # Clean NaNs
    combined = combined.dropna()
    
    print(f"Training Data Shape: {combined.shape}")
    print(f"Num Predictors: {len(potential_predictors)}")
    if len(potential_predictors) == 0:
        print("ERROR: No predictors found!")
        return

    # Check for NaNs/Infs
    if combined[potential_predictors].isnull().values.any():
        print("ERROR: NaNs in predictors!")
        print(combined[potential_predictors].isnull().sum()[combined[potential_predictors].isnull().sum() > 0])
        return
        
    if np.isinf(combined[potential_predictors]).values.any():
        print("ERROR: Infs in predictors!")
        return

    # Check dtypes
    # print(combined[potential_predictors].dtypes)
    non_numeric = combined[potential_predictors].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"ERROR: Non-numeric predictors: {non_numeric}")
        return

    # Debug: Small run
    # combined = combined.head(2000)

    # Scale
    print("Scaling...")
    # Keep raw for final training
    combined_raw = combined.copy()
    
    scaler_full = MinMaxScaler()
    combined[potential_predictors] = scaler_full.fit_transform(combined[potential_predictors])
    
    # SFS
    rr = RidgeClassifier(alpha=1)
    split = TimeSeriesSplit(n_splits=3)
    sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)
    
    print("Running SFS (this takes time)...")
    sfs.fit(combined[potential_predictors], combined["target"])
    
    selected_predictors = list(np.array(potential_predictors)[sfs.get_support()])
    print(f"Selected Predictors: {selected_predictors}")
    
    # Retrain Scaler and Model on Selected ONLY
    print("Retraining Final Model & Scaler on Selected Features...")
    final_scaler = MinMaxScaler()
    X_selected = final_scaler.fit_transform(combined_raw[selected_predictors])
    
    rr.fit(X_selected, combined["target"])
    
    # Save
    print("Saving Artifacts...")
    joblib.dump(rr, f"{MODEL_DIR}/model_v3.pkl")
    joblib.dump(final_scaler, f"{MODEL_DIR}/scaler_v3.pkl")
    joblib.dump(selected_predictors, f"{MODEL_DIR}/predictors_v3.pkl")
    
    # Accuracy check
    preds = rr.predict(X_selected)
    score = accuracy_score(combined["target"], preds)
    print(f"Training Accuracy: {score:.4f}")

if __name__ == "__main__":
    main()
