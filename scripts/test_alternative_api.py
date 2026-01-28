"""
Test script for balldontlie.io API
Tests basic connectivity, game fetching, and stat availability

SETUP:
1. Create free account at https://app.balldontlie.io
2. Get your API key from Account Settings
3. Set environment variable: set BALLDONTLIE_API_KEY=your_key_here
4. Run: python scripts/test_alternative_api.py
"""

import requests
import json
import os
from datetime import datetime, timedelta

# balldontlie.io API base URL
BASE_URL = "https://api.balldontlie.io/v1"

# Get API key from environment variable
API_KEY = os.getenv('BALLDONTLIE_API_KEY')

if not API_KEY:
    print("=" * 60)
    print("ERROR: BALLDONTLIE_API_KEY environment variable not set")
    print("=" * 60)
    print("\nTo get your free API key:")
    print("1. Go to https://app.balldontlie.io")
    print("2. Create a free account")
    print("3. Get your API key from Account Settings")
    print("4. Set environment variable:")
    print("   Windows: set BALLDONTLIE_API_KEY=your_key_here")
    print("   Linux/Mac: export BALLDONTLIE_API_KEY=your_key_here")
    print("\nThen run this script again.")
    exit(1)

# Headers with API key
HEADERS = {
    "Authorization": API_KEY
}

def test_basic_connection():
    """Test basic API connectivity"""
    print("=" * 60)
    print("TEST 1: Basic API Connection")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/teams", headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        teams = response.json()
        print(f"✓ Successfully connected to API")
        print(f"✓ Found {len(teams['data'])} teams")
        print(f"✓ Sample team: {teams['data'][0]['full_name']}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_get_games():
    """Test fetching games for a specific date"""
    print("\n" + "=" * 60)
    print("TEST 2: Fetch Games for Specific Date")
    print("=" * 60)
    
    # Test with a date that should have games (yesterday)
    test_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Testing date: {test_date}")
    
    try:
        params = {
            "start_date": test_date,
            "end_date": test_date
        }
        response = requests.get(f"{BASE_URL}/games", params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        games = response.json()
        game_count = len(games['data'])
        
        print(f"✓ Successfully fetched games")
        print(f"✓ Found {game_count} games")
        
        if game_count > 0:
            sample_game = games['data'][0]
            print(f"\nSample game:")
            print(f"  {sample_game['home_team']['full_name']} vs {sample_game['visitor_team']['full_name']}")
            print(f"  Score: {sample_game['home_team_score']} - {sample_game['visitor_team_score']}")
            print(f"  Status: {sample_game['status']}")
        
        return games['data']
    except Exception as e:
        print(f"✗ Failed to fetch games: {e}")
        return None

def test_get_game_stats(game_id):
    """Test fetching detailed stats for a specific game"""
    print("\n" + "=" * 60)
    print("TEST 3: Fetch Detailed Game Stats")
    print("=" * 60)
    print(f"Testing game ID: {game_id}")
    
    try:
        params = {"game_ids[]": game_id}
        response = requests.get(f"{BASE_URL}/stats", params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        stats = response.json()
        stat_count = len(stats['data'])
        
        print(f"✓ Successfully fetched game stats")
        print(f"✓ Found {stat_count} player stat lines")
        
        if stat_count > 0:
            sample_stat = stats['data'][0]
            print(f"\nSample player stat:")
            print(f"  Player: {sample_stat['player']['first_name']} {sample_stat['player']['last_name']}")
            print(f"  Team: {sample_stat['team']['full_name']}")
            print(f"  Points: {sample_stat['pts']}")
            print(f"  Rebounds: {sample_stat['reb']}")
            print(f"  Assists: {sample_stat['ast']}")
            
            print(f"\nAvailable stat fields:")
            stat_fields = [k for k in sample_stat.keys() if k not in ['player', 'team', 'game']]
            for field in sorted(stat_fields):
                print(f"  - {field}: {sample_stat[field]}")
        
        return stats['data']
    except Exception as e:
        print(f"✗ Failed to fetch game stats: {e}")
        return None

def analyze_stat_availability():
    """Analyze which stats are available vs required for our model"""
    print("\n" + "=" * 60)
    print("TEST 4: Stat Availability Analysis")
    print("=" * 60)
    
    # Stats required by our model (from update_frontend_data.py)
    required_stats = {
        "Traditional Stats": [
            "fg", "fga", "fg3", "fg3a", "ft", "fta",
            "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts"
        ],
        "Advanced Stats": [
            "ts%", "efg%", "fg3a_rate", "fta_rate",
            "oreb%", "dreb%", "reb%", "ast%", "stl%", "blk%", "tov%",
            "ortg", "drtg", "pace", "pie"
        ],
        "Opponent Stats": [
            "All traditional and advanced stats for opponent"
        ]
    }
    
    # Stats available from balldontlie.io (based on API docs)
    available_stats = {
        "Player Stats": [
            "min", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
            "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
            "ast", "stl", "blk", "turnover", "pf", "pts"
        ],
        "Team-Level Stats": "Must be aggregated from player stats",
        "Advanced Stats": "Not directly available - must be calculated"
    }
    
    print("\nREQUIRED STATS:")
    for category, stats in required_stats.items():
        print(f"\n{category}:")
        for stat in stats:
            print(f"  - {stat}")
    
    print("\n\nAVAILABLE FROM balldontlie.io:")
    for category, stats in available_stats.items():
        print(f"\n{category}:")
        if isinstance(stats, list):
            for stat in stats:
                print(f"  - {stat}")
        else:
            print(f"  {stats}")
    
    print("\n" + "=" * 60)
    print("COMPATIBILITY ANALYSIS")
    print("=" * 60)
    print("\n✓ AVAILABLE (with aggregation):")
    print("  - All traditional stats (FG, 3P, FT, rebounds, assists, etc.)")
    print("  - Player-level data can be summed to team totals")
    
    print("\n⚠ REQUIRES CALCULATION:")
    print("  - Advanced stats (TS%, eFG%, ORtg, DRtg, pace, etc.)")
    print("  - These can be calculated from traditional stats")
    print("  - Would need to implement calculation formulas")
    
    print("\n✓ ADVANTAGES:")
    print("  - Free, no authentication required")
    print("  - Cloud-friendly (not blocked in GitHub Actions)")
    print("  - Simple REST API")
    
    print("\n⚠ DISADVANTAGES:")
    print("  - No direct team-level aggregated stats")
    print("  - No pre-calculated advanced metrics")
    print("  - Requires more processing on our end")

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "BALLDONTLIE.IO API TEST SUITE" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Test 1: Basic connection
    if not test_basic_connection():
        print("\n✗ Basic connection failed. Aborting tests.")
        return
    
    # Test 2: Get games
    games = test_get_games()
    if not games or len(games) == 0:
        print("\n⚠ No games found for test date. Trying with a different date...")
        # If no games yesterday, try a few days back
        for days_back in range(2, 8):
            test_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            print(f"\nTrying date: {test_date}")
            params = {"start_date": test_date, "end_date": test_date}
            response = requests.get(f"{BASE_URL}/games", params=params, headers=HEADERS, timeout=10)
            games = response.json()['data']
            if len(games) > 0:
                print(f"✓ Found {len(games)} games")
                break
    
    # Test 3: Get detailed stats for first game
    if games and len(games) > 0:
        test_get_game_stats(games[0]['id'])
    else:
        print("\n⚠ No games available to test stats fetching")
    
    # Test 4: Analyze stat availability
    analyze_stat_availability()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the available stats above")
    print("2. If compatible, test in GitHub Actions")
    print("3. Decide whether to migrate from nba_api")

if __name__ == "__main__":
    main()
