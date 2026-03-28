@echo off
echo ============================================================
echo  OLD TASKS
echo ============================================================
schtasks /query /fo LIST /tn "NHL Nightly Update"
echo.
schtasks /query /fo LIST /tn "NHL Morning Lineup"
echo.
schtasks /query /fo LIST /tn "NHL NST Daily"
echo.
schtasks /query /fo LIST /tn "NHL Game Results Update"
echo.
schtasks /query /fo LIST /tn "NHL Train and Predict"
echo.
schtasks /query /fo LIST /tn "NHL Train and Predict Evening"
echo.
schtasks /query /fo LIST /tn "NHL Pregame Lineup"
echo.
echo ============================================================
echo  NEW TASKS
echo ============================================================
schtasks /query /fo LIST /tn "NHL - 1AM Post-Game Results"
echo.
schtasks /query /fo LIST /tn "NHL - 9AM Morning Predictions"
echo.
schtasks /query /fo LIST /tn "NHL - 1130AM Lineup Update"
echo.
schtasks /query /fo LIST /tn "NHL - 230PM Lineup Update"
echo.
schtasks /query /fo LIST /tn "NHL - 6PM Lineup Update"
echo.
schtasks /query /fo LIST /tn "NHL - 930PM Lineup Update"
