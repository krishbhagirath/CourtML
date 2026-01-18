# 🏀 NBA Match Predictor V5

A production-ready machine learning system that predicts NBA game outcomes with **61.2% accuracy**.
It features a full pipeline: Data Collection -> Training (HistGradientBoosting) -> Live Inference -> React Frontend -> Daily Automation.

## 🚀 Features

- **Advanced Modeling**: Uses `HistGradientBoostingClassifier` with "Matchup Merge" architecture (comparing Team Form vs Opponent Form).
- **Live Predictions**: Fetches today's games via `nba_api`, processes stats in real-time, and predicts winners with confidence scores.
- **Automated Pipeline**: A GitHub Action (`.github/workflows/daily_prediction.yml`) runs every morning at 6 AM ET to generate new predictions.
- **React Frontend**: A clean UI to view today's games and the model's picks.

## 📊 Performance

- **Model**: HistGradientBoostingClassifier
- **Accuracy**: **61.2%** (Verified on 2022-2025 seasons)
- **Key Features**: 50 selected predictors including Rolling Advanced Stats (`orb%`, `drtg`, etc.) and "Matchup" differentials.

## 🛠 Tech Stack

- **ML/Backend:** Python, scikit-learn, pandas, numpy, nba_api
- **Frontend:** React.js, CSS Modules
- **CI/CD:** GitHub Actions (Daily Cron Job)

## 📁 Project Structure

```
nba-match-predictor/
├── predictors/
│   └── predictor_v5.ipynb      # Main training notebook (Analysis & Retraining)
├── models/
│   └── hist_gbm_v5/            # Serialized Model, Scaler, and Predictor list
├── scripts/
│   └── predict_v5.py           # PRODUCTION SCRIPT: Generates today's predictions
├── frontend/                   # React Application
│   ├── public/data/            # Contains schedule and generated predictions.json
│   └── src/                    # Frontend source
├── data/                       # Raw training data (gitignored)
└── .github/workflows/          # Automation configuration
```

## 🏃 How to Run

### 1. Generate Live Predictions
To run the prediction system locally:
```bash
pip install -r requirements.txt
python scripts/predict_v5.py
```
This will fetch today's games and save the results to `frontend/public/data/predictions.json`.

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

## 🤖 Automation
The project is configured to run automatically via GitHub Actions.
- **Schedule**: Every day at 11:00 UTC (6:00 AM ET).
- **Action**: Runs `predict_v5.py`, commits the new `predictions.json`, and pushes to the repo.
- **Deploy**: Vercel (linked to the repo) automatically deploys the updated frontend.

## 📜 License
MIT License.
