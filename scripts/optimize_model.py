import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def optim():
    # Load data
    print("Loading data...")
    data = pd.read_csv("data/nba_games_raw.csv", index_col=0)
    
    # Sort and reset
    data = data.sort_values("date")
    data = data.reset_index(drop=True)
    
    # Cleanup unnecessary columns if they exist
    for col in ["mp.1", "mp_opp.1", "index_opp"]:
        if col in data.columns:
            del data[col]
            
    # Add target
    def add_target(team):
        team["target"] = team["won"].shift(-1)
        return team
    
    data = data.groupby("team", group_keys=False).apply(add_target)
    
    # Handle nulls/types
    data.loc[pd.isnull(data["target"]), "target"] = 2
    data["target"] = data["target"].astype(int, errors="ignore")
    
    # Remove nulls (simplified from notebook logic)
    nulls = pd.isnull(data).sum()
    nulls = nulls[nulls > 0]
    valid_columns = data.columns[~data.columns.isin(nulls.index)]
    data = data[valid_columns].copy()
    
    # === NEW CLEANING STEP ===
    # Remove 'mp' and other leaks/irrelevant columns
    # 'won' is the current game result, 'target' is next game result.
    # We want to predict 'target'. 
    # 'season', 'date', 'team', 'team_opp' are metadata.
    removed_columns = ["season", "date", "won", "target", "team", "team_opp", "home_opp", "mp", "mp_opp"]
    
    # Also ensure we don't include string columns or other leaks
    selected_columns = data.columns[~data.columns.isin(removed_columns)]
    
    # Scale data
    print("Scaling data...")
    scaler = MinMaxScaler()
    data[selected_columns] = scaler.fit_transform(data[selected_columns])
    
    # Initialize Model
    rr = RidgeClassifier(alpha=0.1)
    split = TimeSeriesSplit(n_splits=3)
    
    # SFS
    print("Running SFS (this may take a while)...")
    sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)
    sfs.fit(data[selected_columns], data["target"])
    
    # Get predictors
    predictors = list(selected_columns[sfs.get_support()])
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

    predictions = backtest(data, rr, predictors)
    from sklearn.metrics import accuracy_score
    # Filter out 2s (draws/missing) if any? Target=2 was used for last game.
    # The backtest function just concats.
    # The original notebook didn't filter 2 for accuracy check explicitly in the snippet I saw, 
    # but let's check accuracy where actual != 2
    predictions = predictions[predictions["actual"] != 2]
    
    score = accuracy_score(predictions["actual"], predictions["prediction"])
    print(f"Accuracy: {score}")
    
    # Save artifacts for test.py
    # We need the scaler, the model, and the predictors list potentially?
    # Or just the predictors list if test.py fetches stats and passes them to a loaded model.
    # The original notebook saved 'models/ridge_regression/model_unoptimized.pkl'
    # I'll save new ones.
    if not os.path.exists("models/ridge_regression"):
        os.makedirs("models/ridge_regression")
        
    joblib.dump(rr, "models/ridge_regression/model_optimized.pkl")
    joblib.dump(predictors, "models/ridge_regression/predictors_optimized.pkl") # Save list
    # We might need to save the scaler too if we validly want to use the model on new data.
    joblib.dump(scaler, "models/ridge_regression/scaler_optimized.pkl") # This might be tricky if columns change orders, but ok for now.
    
    print("Optimization complete.")
    
if __name__ == "__main__":
    optim()
