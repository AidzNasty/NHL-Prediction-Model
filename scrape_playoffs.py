"""
scrape_playoffs.py
-------------------
Adds NHL playoff games (schedule + final score + winner + OT flag) to the
Games table for a given season, using the NHL API.

Source of truth is the NHL API:
  - Each series' games:  /v1/schedule/playoff-series/{seasonCode}/{letter}/
  - Date + OT status:    /v1/gamecenter/{gameId}/boxscore
(The series endpoint omits game dates, so the boxscore is used for those.)

Series letters run a.. through the bracket (8 first-round + 4 + 2 + 1 = 15,
letters a-o). The script probes letters until they stop returning games, so it
adapts to however far the playoffs have progressed.

Idempotent: a game already in the DB (matched by matchup + final score) is
skipped, so this is safe to re-run as series play out.

GameStats and PlayerGameLog for the added games are handled by the dedicated
scrapers afterward:
    python scrape_gamestats.py    --season 2025-26
    python scrape_playergamelog.py --season 2025-26

Usage:
    python scrape_playoffs.py --season 2025-26
    python scrape_playoffs.py --season 2024-25
"""

import os
import sys
import time
import string
import argparse
import requests
import duckdb
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

if not TOKEN:
    raise ValueError("MOTHERDUCK_TOKEN not found in .env file")

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Season label -> NHL API season code
SEASON_CODE = {
    "2025-26": "20252026",
    "2024-25": "20242025",
    "2023-24": "20232024",
}

parser = argparse.ArgumentParser()
parser.add_argument("--season", default="2025-26", help="Season e.g. 2025-26")
args = parser.parse_args()

if args.season not in SEASON_CODE:
    print(f"Unknown season: {args.season} (known: {', '.join(SEASON_CODE)})")
    sys.exit(1)
code = SEASON_CODE[args.season]

print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

# Team abbreviation -> TeamID (NHL API abbrevs match the DB's)
abbr2id = {r[0]: r[1] for r in con.execute("SELECT Abbreviation, TeamID FROM Teams").fetchall()}

# Existing playoff games for this season, keyed by matchup + score (for idempotency)
existing = set()
for h, a, hs, as_ in con.execute("""
    SELECT ha.Abbreviation, aa.Abbreviation, g.HomeScore, g.AwayScore
    FROM Games g
    JOIN Teams ha ON g.HomeTeamID = ha.TeamID
    JOIN Teams aa ON g.AwayTeamID = aa.TeamID
    WHERE g.GameType = 'Playoffs' AND g.Season = ?
""", [args.season]).fetchall():
    existing.add((h, a, hs, as_))

next_id = con.execute("SELECT COALESCE(MAX(GameID), 0) FROM Games").fetchone()[0] + 1


def get_json(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code == 429:
        print("  [429] rate limited — sleeping 30s...")
        time.sleep(30)
        r = requests.get(url, headers=HEADERS, timeout=20)
    return r.json() if r.status_code == 200 else None


inserted = skipped = unplayed = 0

for letter in string.ascii_lowercase:  # a..z (bracket only uses a-o)
    series = get_json(f"https://api-web.nhle.com/v1/schedule/playoff-series/{code}/{letter}/")
    games = (series or {}).get("games", [])
    if not games:
        # No series with this letter — assume we've passed the end of the bracket.
        if letter > "o":
            break
        continue

    a0 = games[0].get("awayTeam", {}).get("abbrev", "?")
    h0 = games[0].get("homeTeam", {}).get("abbrev", "?")
    print(f"Series {letter.upper()}  {a0} vs {h0}  ({len(games)} games)")

    for g in games:
        gid = g.get("id")
        box = get_json(f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore")
        time.sleep(0.3)
        if not box:
            continue

        home_abbr = box.get("homeTeam", {}).get("abbrev")
        away_abbr = box.get("awayTeam", {}).get("abbrev")
        home_score = box.get("homeTeam", {}).get("score")
        away_score = box.get("awayTeam", {}).get("score")
        game_date  = box.get("gameDate")

        # Skip games that haven't been played yet (no final score)
        if home_score is None or away_score is None:
            unplayed += 1
            continue

        if (home_abbr, away_abbr, home_score, away_score) in existing:
            skipped += 1
            continue

        home_id = abbr2id.get(home_abbr)
        away_id = abbr2id.get(away_abbr)
        if not home_id or not away_id:
            print(f"  [NO TEAM] {away_abbr} @ {home_abbr}")
            continue

        winner_id = home_id if home_score > away_score else away_id
        is_ot = box.get("gameOutcome", {}).get("lastPeriodType", "REG") != "REG"

        con.execute("""
            INSERT INTO Games (
                GameID, HomeTeamID, AwayTeamID, GameDate, Season,
                GameType, HomeScore, AwayScore, WinnerTeamID, OvertimeFlag
            ) VALUES (?, ?, ?, ?, ?, 'Playoffs', ?, ?, ?, ?)
        """, [next_id, home_id, away_id, game_date, args.season,
              home_score, away_score, winner_id, is_ot])
        existing.add((home_abbr, away_abbr, home_score, away_score))
        next_id += 1
        inserted += 1
        ot_str = " (OT)" if is_ot else ""
        print(f"  + {game_date} {away_abbr} {away_score} @ {home_abbr} {home_score}{ot_str}")

    time.sleep(0.3)

total = con.execute(
    "SELECT COUNT(*) FROM Games WHERE GameType='Playoffs' AND Season=?", [args.season]
).fetchone()[0]
print(f"\n{'='*55}")
print(f"  Inserted:  {inserted}")
print(f"  Skipped:   {skipped} (already in DB)")
print(f"  Unplayed:  {unplayed} (no final score yet)")
print(f"  Total {args.season} playoff games in DB: {total}")
print(f"{'='*55}")
con.close()
print("\nDone! Run scrape_gamestats.py and scrape_playergamelog.py to fill stats.")
