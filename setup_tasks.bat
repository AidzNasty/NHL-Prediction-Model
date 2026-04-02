@echo off
REM ============================================================
REM  NHL Task Scheduler Setup
REM  Removes old tasks, creates new optimized schedule
REM ============================================================
set NHLD=C:\Users\PC\OneDrive - Suffolk University\Prediction Models\NHL

echo Removing old tasks...
schtasks /delete /tn "NHL Nightly Update"          /f 2>nul
schtasks /delete /tn "NHL Morning Lineup"          /f 2>nul
schtasks /delete /tn "NHL Game Results Update"     /f 2>nul
schtasks /delete /tn "NHL Train and Predict"       /f 2>nul
schtasks /delete /tn "NHL Train and Predict Evening" /f 2>nul
schtasks /delete /tn "NHL Pregame Lineup"          /f 2>nul
schtasks /delete /tn "NHL - 1AM Post-Game Results" /f 2>nul
schtasks /delete /tn "NHL - 9AM Morning Predictions" /f 2>nul
schtasks /delete /tn "NHL - 1130AM Lineup Update"  /f 2>nul
schtasks /delete /tn "NHL - 230PM Lineup Update"   /f 2>nul
schtasks /delete /tn "NHL - 6PM Lineup Update"     /f 2>nul
schtasks /delete /tn "NHL - 930PM Lineup Update"   /f 2>nul

echo Creating new tasks...

REM 10:00 AM — Scrape last night's results (HR posts by ~10am)
schtasks /create /tn "NHL - 1000AM Results Scrape" /tr "\"%NHLD%\pipeline_results.bat\"" /sc daily /st 10:00 /f

REM 11:00 AM — Lineup + predict after results are processed
schtasks /create /tn "NHL - 1100AM Lineup Predict" /tr "\"%NHLD%\pipeline_predict.bat\"" /sc daily /st 11:00 /f

REM 12:00 PM — Initial predictions after results are in DB
schtasks /create /tn "NHL - 1200PM Predict"        /tr "\"%NHLD%\pipeline_predict.bat\"" /sc daily /st 12:00 /f

REM 2:00 PM — Pre-afternoon game refresh (3pm weekend games)
schtasks /create /tn "NHL - 200PM Lineup Predict"  /tr "\"%NHLD%\pipeline_predict.bat\"" /sc daily /st 14:00 /f

REM 5:30 PM — Pre-evening game refresh (7pm ET weekday games)
schtasks /create /tn "NHL - 530PM Lineup Predict"  /tr "\"%NHLD%\pipeline_predict.bat\"" /sc daily /st 17:30 /f

REM 9:00 PM — Pre-west coast game refresh (10pm ET games)
schtasks /create /tn "NHL - 900PM Lineup Predict"  /tr "\"%NHLD%\pipeline_predict.bat\"" /sc daily /st 21:00 /f

echo.
echo Done. Current NHL tasks:
schtasks /query /fo TABLE /nh 2>nul | findstr /i "NHL"
