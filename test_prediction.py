"""
Quick test to verify the new keyDifferences output
"""
import sys
sys.path.append('.')

from scripts.update_frontend_data import predict_matchup
import joblib

# Load model, scaler, predictors, feature importance
model = joblib.load('models/hist_gbm_v5/model_v5.pkl')
scaler = joblib.load('models/hist_gbm_v5/scaler_v5.pkl')
predictors = joblib.load('models/hist_gbm_v5/predictors_v5.pkl')

try:
    feature_importances = joblib.load('models/hist_gbm_v5/feature_importance_v5.pkl')
    print("Loaded feature importance scores")
except FileNotFoundError:
    feature_importances = None
    print("No feature importance file found, using None")

# Test with Cavaliers vs Lakers (IDs from today.json)
home_id = 1610612739  # Cavaliers
away_id = 1610612747  # Lakers

print("Testing predict_matchup with new keyDifferences format...")
result = predict_matchup(home_id, away_id, model, scaler, predictors, feature_importances)

if result:
    print("\n✅ SUCCESS!")
    print(f"Winner: {result['winner']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"\nTop {len(result['keyDifferences'])} Differentiating Features:")
    for i, feat in enumerate(result['keyDifferences'], 1):
        advantage = "Home" if feat['homeAdvantage'] else "Away"
        print(f"{i}. {feat['name']}: Home={feat['homeValue']}, Away={feat['awayValue']} (Diff={feat['difference']}, {advantage} advantage)")
else:
    print("❌ FAILED - No result returned")
