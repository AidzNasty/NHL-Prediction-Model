@echo off
REM ============================================================
REM  NHL Pipeline — 9:00 AM  (morning predictions)
REM  Pulls morning skate lineups, injury news, generates
REM  today's predictions before first puck drop
REM ============================================================
cd /d "C:\Users\PC\OneDrive - Suffolk University\Prediction Models\NHL"
set PYTHON=C:\Users\PC\anaconda3\python.exe

echo [%date% %time%] Starting morning pipeline...

echo Scraping Daily Faceoff injury and lineup news...
%PYTHON% scrape_dailyfaceoff.py

echo Scraping NHL API morning skate lineups...
%PYTHON% scrape_nhl_lineups.py

echo Training model and generating today's predictions...
%PYTHON% train_and_predict.py

echo [%date% %time%] Morning pipeline complete.
