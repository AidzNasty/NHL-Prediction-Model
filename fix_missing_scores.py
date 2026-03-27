"""
fix_missing_scores.py
----------------------
One-time fix: updates Games.HomeScore/AwayScore for all games
that have NULL scores but do have GameStats rows.

Scrapes the scorebox from Hockey Reference for each missing game.

Run once to catch up, then scrape_gamestats.py will handle it going forward.

Usage: python fix_missing_scores.py
"""

import os
import time
import requests
import duckdb
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.hockey-reference.com/",
}

# DB abbrev -> HR abbrev
DB_TO_HR = {
    "ANA":"ANA","BOS":"BOS","BUF":"BUF","CGY":"CGY","CAR":"CAR",
    "CHI":"CHI","COL":"COL","CBJ":"CBJ","DAL":"DAL","DET":"DET",
    "EDM":"EDM","FLA":"FLA","L.A":"LAK","MIN":"MIN","MTL":"MTL",
    "NSH":"NSH","N.J":"NJD","NYI":"NYI","NYR":"NYR","OTT":"OTT",
    "PHI":"PHI","PIT":"PIT","S.J":"SJS","SEA":"SEA","STL":"STL",
    "T.B":"TBL","TOR":"TOR","UTA":"UTA","VAN":"VAN","VGK":"VEG",
    "WSH":"WSH","WPG":"WPG",
}

print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

# Find all games with NULL scores
games = con.execute("""
    SELECT g.GameID, g.GameDate, g.Season,
           g.HomeTeamID, g.AwayTeamID,
           ht.Abbreviation AS HomeAbbrev,
           awt.Abbreviation AS AwayAbbrev
    FROM Games g
    JOIN Teams ht  ON g.HomeTeamID = ht.TeamID
    JOIN Teams awt ON g.AwayTeamID = awt.TeamID
    WHERE g.HomeScore IS NULL
      AND g.GameDate < CURRENT_DATE
      AND g.Season IN ('2024-25', '2025-26')
    ORDER BY g.GameDate ASC
""").fetchall()

print(f"Games with missing scores: {len(games)}")
if not games:
    print("All scores already populated!")
    con.close()
    exit()

updated = 0
failed  = 0

for game_id, game_date, season, home_id, away_id, home_abbrev, away_abbrev in games:
    hr_home = DB_TO_HR.get(home_abbrev, home_abbrev)
    date_str = str(game_date).replace("-", "")
    url = f"https://www.hockey-reference.com/boxscores/{date_str}0{hr_home}.html"

    time.sleep(4)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 404:
            print(f"  [404] {date_str} {away_abbrev}@{home_abbrev}")
            failed += 1
            continue
        if resp.status_code == 429:
            print(f"  [429] Rate limited — sleeping 60s...")
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            failed += 1
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        scorebox = soup.find("div", {"class": "scorebox"})
        if not scorebox:
            failed += 1
            continue

        # Get scores — first div = away, second = home
        score_divs = scorebox.find_all("div", {"class": "score"})
        if len(score_divs) < 2:
            failed += 1
            continue

        away_score = int(score_divs[0].get_text(strip=True))
        home_score = int(score_divs[1].get_text(strip=True))
        winner_id  = home_id if home_score > away_score else away_id

        # Check OT
        is_ot = False
        meta = scorebox.find("div", {"class": "scorebox_meta"})
        if meta and any(x in meta.get_text().lower()
                        for x in ["overtime", "shootout"]):
            is_ot = True

        con.execute(f"""
            UPDATE Games SET
                HomeScore    = {home_score},
                AwayScore    = {away_score},
                WinnerTeamID = {winner_id},
                OvertimeFlag = {str(is_ot).upper()}
            WHERE GameID = {game_id}
        """)

        print(f"  OK {game_date} {away_abbrev} {away_score} - {home_score} {home_abbrev}")
        updated += 1

    except Exception as e:
        print(f"  [ERROR] {game_id}: {e}")
        failed += 1

print(f"\n{'='*50}")
print(f"  Updated: {updated}")
print(f"  Failed:  {failed}")
print(f"{'='*50}")
con.close()
print("\nDone!")
