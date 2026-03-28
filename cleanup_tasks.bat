@echo off
schtasks /delete /tn "NHL Nightly Update"            /f
schtasks /delete /tn "NHL Morning Lineup"            /f
schtasks /delete /tn "NHL NST Daily"                 /f
schtasks /delete /tn "NHL Game Results Update"       /f
schtasks /delete /tn "NHL Train and Predict"         /f
schtasks /delete /tn "NHL Train and Predict Evening" /f
schtasks /delete /tn "NHL Pregame Lineup"            /f
echo.
echo Remaining NHL tasks:
schtasks /query /fo TABLE /nh | findstr /i "NHL"
