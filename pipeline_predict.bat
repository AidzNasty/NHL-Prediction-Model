@echo off
REM ============================================================
REM  NHL Pipeline — Lineup + Predict
REM  Pulls latest confirmed lineups and injury news,
REM  then re-generates predictions with fresh data.
REM  Safe to run multiple times — only updates games not yet played.
REM
REM  Scheduled at:
REM    12:00 PM  — after results are in, initial predictions
REM    2:00 PM   — before afternoon/early games
REM    5:30 PM   — before 7pm ET games (main weekday window)
REM    9:00 PM   — before 10pm ET west coast games
REM ============================================================
cd /d "C:\Users\PC\OneDrive - Suffolk University\Prediction Models\NHL"
set PYTHON=C:\Users\PC\anaconda3\python.exe

echo [%date% %time%] Starting lineup + predict pipeline...

echo Scraping confirmed NHL lineups from NHL API...
%PYTHON% scrape_nhl_lineups.py

echo Scraping Daily Faceoff for latest injury/status news...
%PYTHON% scrape_dailyfaceoff.py

echo Re-generating predictions with latest lineup data...
%PYTHON% train_and_predict.py

echo [%date% %time%] Lineup + predict pipeline complete.
