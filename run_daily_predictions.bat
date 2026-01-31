@echo off
SETLOCAL
REM NBA Prediction Automation - Daily Runner
REM Runs at 6 AM daily via Windows Task Scheduler

SET PROJECT_DIR=C:\Users\seema\Personal Projects\NBA Match Predictor
SET PYTHON_EXE=C:\Python313\python.exe
SET GIT_EXE="C:\Program Files\Git\cmd\git.exe"
SET LOG_FILE=%PROJECT_DIR%\automation_log.txt
SET PYTHONUNBUFFERED=1
SET PYTHONUTF8=1

echo ============================================================ >> "%LOG_FILE%"
echo NBA PREDICTION AUTOMATION - %date% %time% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

REM Navigate to project directory
cd /d "%PROJECT_DIR%"

echo. >> "%LOG_FILE%"
echo [1/4] Running prediction script... >> "%LOG_FILE%"
"%PYTHON_EXE%" scripts\update_frontend_data.py >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Prediction script failed! >> "%LOG_FILE%"
    exit /b 1
)

echo. >> "%LOG_FILE%"
echo [2/4] Adding changed files... >> "%LOG_FILE%"
%GIT_EXE% add data\predictions_history.json >> "%LOG_FILE%" 2>&1
%GIT_EXE% add frontend\public\data\today.json >> "%LOG_FILE%" 2>&1
%GIT_EXE% add frontend\public\data\lastweek.json >> "%LOG_FILE%" 2>&1

echo. >> "%LOG_FILE%"
echo [3/4] Committing changes... >> "%LOG_FILE%"
%GIT_EXE% commit --author="NBA Bot <nba-bot@courtml.local>" -m "Update daily predictions [automated] - %date% %time%" >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Commit successful! >> "%LOG_FILE%"
) else (
    echo No changes to commit >> "%LOG_FILE%"
)

echo. >> "%LOG_FILE%"
echo [4/4] Pushing to GitHub... >> "%LOG_FILE%"
%GIT_EXE% push origin main >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Push failed - check network connection >> "%LOG_FILE%"
    exit /b 1
)

echo. >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo COMPLETE! >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
ENDLOCAL
