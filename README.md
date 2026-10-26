# NBA Match Predictor V5

A production-ready machine learning system that predicts NBA game outcomes.
It features a full pipeline: Data Collection -> Training (HistGradientBoosting) -> Live Inference -> React Frontend -> Daily Automation.

## Features

- **Advanced Modeling**: Uses `HistGradientBoostingClassifier` with "Matchup Merge" architecture (comparing Team Form vs Opponent Form).
- **Live Predictions**: Fetches today's games via `nba_api`, processes stats in real-time, and predicts winners with confidence scores.
- **Automated Pipeline**: A GitHub Action (`.github/workflows/daily_prediction.yml`) runs every morning at 6 AM ET to generate new predictions.
- **React Frontend**: A clean UI to view today's games and the model's picks.

## Performance

- **Model**: HistGradientBoostingClassifier
- **Key Features**: 50 selected predictors including Rolling Advanced Stats (`orb%`, `drtg`, etc.) and "Matchup" differentials.

## Tech Stack

- **ML/Backend:** Python, scikit-learn, pandas, numpy, nba_api
- **Frontend:** React.js, CSS Modules
- **CI/CD:** GitHub Actions (Daily Cron Job)

## Project Structure

```
nba-match-predictor/
├── predictors/
│   └── predictor_v5.ipynb      # Main training notebook (Analysis & Retraining)
├── models/
│   └── hist_gbm_v5/            # Serialized Model, Scaler, and Predictor list
├── scripts/
│   ├── update_frontend_data.py # PRODUCTION SCRIPT: predicts today, grades past picks
│   └── evaluate_model.py       # Offline check: held-out accuracy, calibration, feature importance
├── frontend/                   # React Application
│   ├── public/data/            # today.json, lastweek.json, performance.json (generated)
│   └── src/                    # Frontend source
├── data/                       # Raw training data (gitignored)
└── .github/workflows/          # Automation configuration
```

## How to Run

### 1. Generate Live Predictions
To run the prediction system locally:
```bash
pip install -r requirements.txt
python scripts/update_frontend_data.py            # or --date YYYY-MM-DD
```
This fetches the league game log and today's schedule from `nba_api`, predicts each game, grades past
predictions, and writes `today.json`, `lastweek.json` and `performance.json` to `frontend/public/data/`.
Box scores are cached in `data/boxscore_cache/` (gitignored), so after the first run only the previous
night's games are downloaded.

To re-check the model offline (needs `data/nba_games_raw.csv`):
```bash
python scripts/evaluate_model.py
```
On the held-out 2021–22 season it picks 61.9% of winners (always picking the home team: 54.7%).

### 2. Run the Frontend
```bash
cd frontend
npm install
npm start
```
Open [http://localhost:3000](http://localhost:3000) to see the dashboard.

### 3. Retrain the Model
Open `predictors/predictor_v5.ipynb` in Jupyter. This notebook contains the full pipeline to:
1. Load `data/nba_games_raw.csv`
2. Clean and Compute Rolling Averages
3. Train the HistGradientBoosting model
4. Save artifacts to `models/`

## Automation
The project is configured to run automatically via GitHub Actions.
- **Schedule**: Daily at 6:00 AM via Windows Task Scheduler (`run_daily_predictions.bat`). The GitHub Action is disabled.
- **Action**: Runs `update_frontend_data.py`, commits the updated JSON files, and pushes to the repo.
- **Deploy**: Vercel (linked to the repo) automatically deploys the updated frontend.

## License
MIT License.
