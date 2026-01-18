import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import numpy as np

def optim_v2():
    print("Loading data...")
    data = pd.read_csv("data/nba_games_raw.csv", index_col=0)
    data = data.sort_values("date").reset_index(drop=True)
    
    # Cleanup
    for col in ["mp.1", "mp_opp.1", "index_opp"]:
        if col in data.columns:
            del data[col]
            
    # Add target (Next game win) BEFORE rolling? 
    # The 'won' column is current game. 
    # Rolling averages should be based on PAST games to predict NEXT game.
    # So we calculate rolling avg of 'won' (recent form) and stats.
    # THEN we shift 'won' to get 'target' (next game result).
    # Wait, 'target' is "outcome of NEXT match".
    # So for row i, target is row i+1.
    # Predictors for row i should be rolling average of i, i-1, i-2...
    # Yes.
    
    def add_target(team):
        team["target"] = team["won"].shift(-1)
        return team

    data = data.groupby("team", group_keys=False).apply(add_target)
    data.loc[pd.isnull(data["target"]), "target"] = 2
    data["target"] = data["target"].astype(int, errors="ignore")
    
    # Simple clean
    nulls = pd.isnull(data).sum()
    nulls = nulls[nulls > 0]
    valid_columns = data.columns[~data.columns.isin(nulls.index)]
    data = data[valid_columns].copy()
    
    # Define columns to remove from predictors (metadata + leaks)
    # We want to keep 'won' in the rolling average (to see form), but not use it as a raw predictor?
    # Actually, rolling win % is a valid predictor.
    # Raw 'won' of current game is also valid predictor (did they win last game?).
    # But 'target' must be removed. 'mp' must be removed as requested.
    
    removed_columns = ["season", "date", "target", "team", "team_opp", "home_opp", "mp", "mp_opp"]
    selected_columns = data.columns[~data.columns.isin(removed_columns)]
    
    # Rolling Averages
    print("Computing rolling averages...")
    # converting to numeric strictly to avoid errors
    numeric_cols = data[selected_columns].select_dtypes(include=[np.number]).columns
    
    # We need to preserve 'team', 'season' for grouping
    # And 'target', 'date' for splitting/training
    
    def find_team_averages(team):
        # Rolling on numeric columns only
        # Closed='left' means for time t, window is [t-10, t)? 
        # No, standard rolling includes current row.
        # Since we are predicting 'target' (next game) using current row's stats,
        # standard rolling(10) means average of current game + 9 previous.
        # This is correct.
        rolling = team[numeric_cols].rolling(10).mean()
        return rolling

    data_rolling = data.groupby(["team", "season"], group_keys=False).apply(find_team_averages)
    
    # Rename columns
    rolling_cols = [f"{col}_10" for col in data_rolling.columns]
    data_rolling.columns = rolling_cols
    
    # Concatenate? Or just use rolling?
    # User said "pull 2-3 week averages... and format it...".
    # This implies using ONLY rolling averages (and maybe current scalars like Home/Away?).
    # If we mix raw + rolling, we have huge feature set.
    # Let's stick to Rolling + Home/Away (from original) + Target (from original).
    # 'home' was in selected_columns (numeric). So it got rolled?
    # Rolling average of 'home' (0 or 1) tells you how many home games recently. Might be useful?
    # But usually you want "Is the NEXT game home/away?"
    # Converting 'home' to rolling is weird for 'next' game prediction context if we want to know if *current* row is home.
    # But 'data_rolling' replaces 'data' values.
    # Row i in data_rolling is avg of i, i-1...
    # Row i's target is result of i+1.
    # We want to predict i+1 result.
    # We traditionally use i's stats.
    # But for Home/Away, we need i+1's location?
    # The 'home' column in `data` row i refers to game i.
    # `target` is game i+1.
    # We need `home_next`?
    # `data["home_next"] = data["home"].shift(-1)`
    # This is important.
    
    data["home_next"] = data.groupby("team", group_keys=False)["home"].shift(-1)
    
    # Combine
    # We use data_rolling as the features.
    # We add `home_next` (raw) and `target` (raw).
    # We also need `season` and `date` for backtesting.
    
    final_data = pd.concat([data_rolling, data[["season", "date", "target", "home_next", "team", "team_opp"]]], axis=1)
    
    # Drop rows with NaN (first 10 games of season, and last game where target/home_next is NaN)
    final_data = final_data.dropna()
    
    # Clean predictors list
    # remove 'season', 'date', 'target', 'team', 'team_opp'
    # 'home_next' should be kept? Yes.
    
    predictors_candidates = list(data_rolling.columns) + ["home_next"]
    
    print(f"Feature count: {len(predictors_candidates)}")
    
    # Scale
    print("Scaling...")
    scaler = MinMaxScaler()
    final_data[predictors_candidates] = scaler.fit_transform(final_data[predictors_candidates])
    
    # SFS
    print("Running SFS (rolling)...")
    rr = RidgeClassifier(alpha=0.1)
    split = TimeSeriesSplit(n_splits=3)
    sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)
    
    # Work with smaller subset to speed up SFS?
    # 130 columns is a lot for SFS.
    # But we want best predictors.
    # We'll run it.
    
    sfs.fit(final_data[predictors_candidates], final_data["target"])
    
    predictors = list(np.array(predictors_candidates)[sfs.get_support()])
    print(f"Selected Predictors: {predictors}")
    
    # Backtest
    print("Backtesting...")
    def backtest(data, model, predictors, start=2, step=1):
        all_predictions = []
        seasons = sorted(data["season"].unique())
        for i in range(start, len(seasons), step):
            season = seasons[i]
            train = data[data["season"] < season]
            test = data[data["season"] == season]
            
            model.fit(train[predictors], train["target"])
            preds = model.predict(test[predictors])
            preds = pd.Series(preds, index=test.index)
            combined = pd.concat([test["target"], preds], axis=1)
            combined.columns = ["actual", "prediction"]
            all_predictions.append(combined)
            
        return pd.concat(all_predictions)

    predictions = backtest(final_data, rr, predictors)
    predictions = predictions[predictions["actual"] != 2]
    from sklearn.metrics import accuracy_score
    score = accuracy_score(predictions["actual"], predictions["prediction"])
    print(f"Accuracy: {score}")
    
    # Save artifacts
    if not os.path.exists("models/ridge_regression"):
        os.makedirs("models/ridge_regression")
        
    joblib.dump(rr, "models/ridge_regression/model_optimized.pkl")
    joblib.dump(predictors, "models/ridge_regression/predictors_optimized.pkl")
    joblib.dump(scaler, "models/ridge_regression/scaler_optimized.pkl")
    
    print("Optimization v2 complete.")

if __name__ == "__main__":
    optim_v2()
