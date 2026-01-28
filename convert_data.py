"""
Quick script to convert existing teamStats format to keyDifferences format
"""
import json

# Load today.json
with open('frontend/public/data/today.json', 'r') as f:
    data = json.load(f)

# Convert each game's prediction
for game in data.get('games', []):
    if 'prediction' in game and 'teamStats' in game['prediction']:
        team_stats = game['prediction']['teamStats']
        home_stats = team_stats.get('homeStats', {})
        away_stats = team_stats.get('awayStats', {})
        labels = team_stats.get('labels', [])
        
        # Map old keys to labels
        key_to_label = {
            'ortg_10': 'Off Rating',
            'drtg_10': 'Def Rating',
            'ts%_10': 'True Shooting %',
            'efg%_10': 'Effective FG %',
            'tov%_10': 'Turnover %',
            'trb%_10': 'Rebound %',
            'ast%_10': 'Assist %'
        }
        
        # Calculate differences
        key_differences = []
        for key, label in key_to_label.items():
            if key in home_stats and key in away_stats:
                home_val = home_stats[key]
                away_val = away_stats[key]
                diff = abs(home_val - away_val)
                
                key_differences.append({
                    'name': label,
                    'homeValue': round(home_val, 1),
                    'awayValue': round(away_val, 1),
                    'difference': round(diff, 1),
                    'homeAdvantage': home_val > away_val
                })
        
        # Sort by difference
        key_differences.sort(key=lambda x: x['difference'], reverse=True)
        
        # Replace teamStats with keyDifferences
        del game['prediction']['teamStats']
        game['prediction']['keyDifferences'] = key_differences

# Save back
with open('frontend/public/data/today.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Successfully converted today.json to keyDifferences format!")
print(f"Updated {len(data.get('games', []))} games")
