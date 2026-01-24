@echo off
REM NBA Prediction Automation - Daily Runner
REM Runs at 6 AM daily via Windows Task Scheduler

echo ============================================================
echo NBA PREDICTION AUTOMATION - %date% %time%
echo ============================================================

REM Navigate to project directory
cd /d "C:\Users\seema\Personal Projects\NBA Match Predictor"

REM Activate virtual environment if you have one (optional)
REM call venv\Scripts\activate

echo.
echo [1/4] Running prediction script...
python scripts\update_frontend_data.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Prediction script failed!
    exit /b 1
)

echo.
echo [2/4] Adding changed files...
git add data/predictions_history.json
git add frontend/public/data/today.json
git add frontend/public/data/lastweek.json

echo.
echo [3/4] Committing changes...
git commit -m "Update daily predictions [automated from local scheduler] - %date%"
if %ERRORLEVEL% EQU 0 (
    echo Commit successful!
) else (
    echo No changes to commit
)

echo.
echo [4/4] Pushing to GitHub...
git push origin main
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Push failed - check network connection
    exit /b 1
)

echo.
echo ============================================================
echo COMPLETE! Predictions updated and pushed to GitHub
echo ============================================================
