"""
Offline validation for the V5 model.

Rebuilds the exact feature pipeline from predictors/predictor_v5.ipynb on
data/nba_games_raw.csv, scores the saved model on the same held-out 15%, and
writes permutation feature importances used to rank "key factors" in the UI.

Usage: python scripts/evaluate_model.py [--skip-importance]
"""
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")

DATA_PATH = "data/nba_games_raw.csv"
MODEL_DIR = "models/hist_gbm_v5"


def build_matchup_frame():
    """Same steps as predictor_v5.ipynb cells 2-5."""
    df = pd.read_csv(DATA_PATH).sort_values("date")
    for c in ["mp", "mp.1", "item"]:
        if c in df.columns:
            del df[c]
    nulls = pd.isnull(df).sum()
    nulls = nulls[nulls > 0]
    df = df[df.columns[~df.columns.isin(nulls.index)]].copy()

    def add_target(team_df):
        team_df["target"] = team_df["won"].shift(-1)
        team_df["home_next"] = team_df["home"].shift(-1)
        team_df["team_opp_next"] = team_df["team_opp"].shift(-1)
        team_df["date_next"] = team_df["date"].shift(-1)
        return team_df

    df = df.groupby("team", group_keys=False).apply(add_target)
    df = df.dropna(subset=["target", "home_next", "team_opp_next", "date_next"])
    df["target"] = df["target"].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in ["season", "target", "won", "team", "team_opp", "date", "home_next", "team_opp_next", "date_next"]:
        if c in numeric_cols:
            numeric_cols.remove(c)

    rolling = df.groupby(["team"], group_keys=False).apply(lambda t: t[numeric_cols].rolling(10).mean())
    rolling.columns = [f"{c}_10" for c in rolling.columns]
    df = pd.concat([df, rolling], axis=1).dropna()

    right_df = df[list(rolling.columns) + ["team", "date_next"]].copy()
    combined = pd.merge(
        df, right_df,
        left_on=["team_opp_next", "date_next"],
        right_on=["team", "date_next"],
        suffixes=("_team", "_opp"),
    ).dropna()
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-importance", action="store_true")
    args = parser.parse_args()

    model = joblib.load(f"{MODEL_DIR}/model_v5.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler_v5.pkl")
    predictors = joblib.load(f"{MODEL_DIR}/predictors_v5.pkl")

    print("Rebuilding training features (takes ~1 min)...")
    combined = build_matchup_frame()
    split_idx = int(len(combined) * 0.85)
    test = combined.iloc[split_idx:]
    print(f"Held-out set: {len(test)} team-games, {test['date_next'].min()} -> {test['date_next'].max()}")

    X = scaler.transform(test[predictors].values)
    y = test["target"].values
    p = model.predict_proba(X)[:, 1]
    print(f"\nTeam-perspective accuracy: {accuracy_score(y, p > 0.5):.4f}  log loss {log_loss(y, p):.4f}")

    # Per-game view: pair each home row with the same game's away row
    t = test.assign(p=p)
    home = t[t["home_next"] == 1][["team_team", "team_opp_next", "date_next", "p", "target"]]
    away = t[t["home_next"] == 0][["team_team", "team_opp_next", "date_next", "p"]]
    games = home.merge(
        away, left_on=["team_team", "team_opp_next", "date_next"],
        right_on=["team_opp_next", "team_team", "date_next"], suffixes=("", "_away"),
    )
    games["p_sym"] = (games["p"] + (1 - games["p_away"])) / 2
    y_g = games["target"]
    print(f"\nPer game (n={len(games)}):")
    print(f"  always pick home      acc {y_g.mean():.4f}")
    print(f"  home-perspective only acc {accuracy_score(y_g, games['p'] > 0.5):.4f}  "
          f"log loss {log_loss(y_g, games['p']):.4f}  brier {brier_score_loss(y_g, games['p']):.4f}")
    print(f"  symmetric (used live) acc {accuracy_score(y_g, games['p_sym'] > 0.5):.4f}  "
          f"log loss {log_loss(y_g, games['p_sym']):.4f}  brier {brier_score_loss(y_g, games['p_sym']):.4f}")

    # Calibration: confidence in the picked side vs how often that side won
    conf = np.maximum(games["p_sym"], 1 - games["p_sym"])
    hit = (games["p_sym"] > 0.5) == (y_g == 1)
    print("\n  Calibration (symmetric):")
    for lo in [0.5, 0.6, 0.7, 0.8]:
        mask = (conf >= lo) & (conf < lo + 0.1 if lo < 0.8 else conf <= 1)
        if mask.sum():
            print(f"    {int(lo*100)}-{int(lo*100)+10 if lo < 0.8 else 100}%: n={mask.sum():4d}  "
                  f"avg conf {conf[mask].mean():.3f}  actual {hit[mask].mean():.3f}")

    if args.skip_importance:
        return
    print("\nComputing permutation importance (log loss)...")
    result = permutation_importance(model, X, y, scoring="neg_log_loss", n_repeats=5, random_state=42, n_jobs=-1)
    importances = {f: float(max(v, 0.0)) for f, v in zip(predictors, result.importances_mean)}
    total = sum(importances.values()) or 1.0
    importances = {f: v / total for f, v in importances.items()}
    joblib.dump(importances, f"{MODEL_DIR}/feature_importance_v5.pkl")
    print("Top 10 features:")
    for f, v in sorted(importances.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {f:24s} {v:.3f}")
    print(f"Saved {MODEL_DIR}/feature_importance_v5.pkl")


if __name__ == "__main__":
    main()
