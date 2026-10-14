"""
seed_season.py
---------------
Loads a full season schedule (preseason + regular season) from the NHL API
into the Games table, and fills in scores for games that have finished.

Safe to run repeatedly — existing games are matched on date/home/away/type,
and only games still missing a score get updated.

Usage:
    python seed_season.py                    # 2026-27
    python seed_season.py --season 2026-27
"""

import os
import sys
import argparse
import requests
from dotenv import load_dotenv
import duckdb

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
load_dotenv()

TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

HEADERS = {"User-Agent": "Mozilla/5.0"}

# NHL API gameType: 1=Preseason, 2=Regular Season, 3=Playoffs (playoffs seeded by seed_playoff_games.py)
GAME_TYPES = {1: "Preseason", 2: "Regular Season"}

parser = argparse.ArgumentParser()
parser.add_argument("--season", default="2026-27", help="Season e.g. 2026-27")
args = parser.parse_args()

season     = args.season
api_season = f"{season[:4]}20{season[5:]}"   # 2026-27 -> 20262027

print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

teams     = con.execute("SELECT TeamID, Abbreviation FROM Teams").fetchall()
by_abbrev = {r[1].upper(): r[0] for r in teams}

# ── Fetch every club's schedule, dedupe by NHL game id ────────
games = {}
for abbrev in sorted(by_abbrev):
    url  = f"https://api-web.nhle.com/v1/club-schedule-season/{abbrev}/{api_season}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        print(f"  [HTTP {resp.status_code}] {abbrev}")
        continue
    for g in resp.json().get("games", []):
        if g.get("gameType") in GAME_TYPES:
            games[g["id"]] = g

print(f"Fetched {len(games)} games for {season}\n")

# ── Insert / update ───────────────────────────────────────────
max_game_id = con.execute("SELECT COALESCE(MAX(GameID), 0) FROM Games").fetchone()[0]

inserted = updated = skipped = errors = 0

for g in sorted(games.values(), key=lambda x: (x["gameDate"], x["id"])):
    game_type = GAME_TYPES[g["gameType"]]
    game_date = g["gameDate"]
    home_ab   = g["homeTeam"]["abbrev"].upper()
    away_ab   = g["awayTeam"]["abbrev"].upper()
    home_id   = by_abbrev.get(home_ab)
    away_id   = by_abbrev.get(away_ab)

    if not home_id or not away_id:
        # Preseason can include non-NHL opponents
        print(f"  [NO TEAM] {game_date} {away_ab} @ {home_ab}")
        errors += 1
        continue

    final = g.get("gameState") in ("OFF", "FINAL")
    if final:
        home_score = g["homeTeam"].get("score")
        away_score = g["awayTeam"].get("score")
        winner_id  = home_id if home_score > away_score else away_id
        ot_flag    = (g.get("gameOutcome") or {}).get("lastPeriodType") in ("OT", "SO")
    else:
        home_score = away_score = winner_id = None
        ot_flag    = False

    existing = con.execute("""
        SELECT GameID, HomeScore FROM Games
        WHERE GameDate = ? AND HomeTeamID = ? AND AwayTeamID = ? AND GameType = ?
    """, [game_date, home_id, away_id, game_type]).fetchone()

    if existing:
        if final and existing[1] is None:
            con.execute("""
                UPDATE Games SET HomeScore = ?, AwayScore = ?, WinnerTeamID = ?, OvertimeFlag = ?
                WHERE GameID = ?
            """, [home_score, away_score, winner_id, ot_flag, existing[0]])
            updated += 1
        else:
            skipped += 1
        continue

    max_game_id += 1
    con.execute("""
        INSERT INTO Games (
            GameID, HomeTeamID, AwayTeamID, GameDate, Season,
            GameType, HomeScore, AwayScore, WinnerTeamID, OvertimeFlag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [max_game_id, home_id, away_id, game_date, season,
          game_type, home_score, away_score, winner_id, ot_flag])
    inserted += 1

summary = con.execute("""
    SELECT GameType, COUNT(*), COUNT(HomeScore) FROM Games
    WHERE Season = ? GROUP BY GameType ORDER BY GameType
""", [season]).fetchall()

print(f"\n{'='*50}")
print(f"  Inserted: {inserted}")
print(f"  Scores updated: {updated}")
print(f"  Skipped:  {skipped} (already up to date)")
print(f"  Errors:   {errors}")
for gt, n, done in summary:
    print(f"  {season} {gt:15} {n:5} games, {done} played")
print(f"{'='*50}")

con.close()
