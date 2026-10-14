@echo off
REM ============================================================
REM  NHL Pipeline — 10:00 AM  (post-game results)
REM  Hockey Reference has last night's box scores by ~10am.
REM  Scrapes scores, player logs, updates accuracy.
REM ============================================================
cd /d "C:\Users\PC\OneDrive - Suffolk University\Prediction Models\NHL"
set PYTHON=C:\Users\PC\anaconda3\python.exe

echo [%date% %time%] Starting results pipeline...

echo Updating 2026-27 schedule and final scores from NHL API...
%PYTHON% seed_season.py

echo Scraping any 2026-27 box scores not yet in DB (catches up missed days)...
%PYTHON% scrape_gamestats.py --season 2026-27

echo Scraping any 2026-27 player game logs not yet in DB...
%PYTHON% scrape_player_gamelogs.py --backfill --season 2026-27

echo Updating player prediction accuracy...
%PYTHON% update_player_accuracy.py

echo Resolving pending team predictions...
%PYTHON% fix_pending_predictions.py

echo Refreshing team standings, player and goalie stats from NHL API...
%PYTHON% update_stats_nhlapi.py

echo Updating calculated fields...
%PYTHON% update_calculated_fields.py

echo [%date% %time%] Results pipeline complete.
