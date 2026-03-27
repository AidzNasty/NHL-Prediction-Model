"""
add_missing_players.py
-----------------------
Finds players on current NHL rosters who are not in our Players table,
and inserts them.

Matching strategy:
  1. Exact normalized full name + team
  2. Normalized last name + team
  If neither matches, the player is truly missing -> INSERT.

Source: NHL API /v1/roster/{ABBREV}/current

Usage: python add_missing_players.py
"""

import os
import time
import unicodedata
import requests
import duckdb
from dotenv import load_dotenv
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

def normalize(s):
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    for k, v in {"ø":"o","Ø":"O","ß":"ss","æ":"ae","Æ":"AE","ð":"d","þ":"th",
                  "\u0142":"l","\u0141":"L"}.items():
        stripped = stripped.replace(k, v)
    return stripped.lower().strip()

def safe_int(val):
    try:
        return int(val) if val is not None else None
    except:
        return None

def safe_str(val):
    return str(val).strip() if val else None

print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

# Load all active players from DB
db_players = con.execute("""
    SELECT p.PlayerID, p.FirstName, p.LastName, t.Abbreviation
    FROM Players p
    JOIN Teams t ON p.TeamID = t.TeamID
    WHERE p.IsActive = TRUE
""").fetchall()

# Build lookups: (norm_full, abbrev) and (norm_last, abbrev)
by_full_team = {}
by_last_team = {}
for pid, first, last, abbrev in db_players:
    norm_full = normalize(f"{first} {last}")
    norm_last = normalize(last)
    by_full_team[(norm_full, abbrev)] = pid
    # setdefault: keep first match for last name (avoids overwriting with different player)
    by_last_team.setdefault((norm_last, abbrev), pid)

# Get all teams
teams = con.execute("SELECT TeamID, TeamName, Abbreviation FROM Teams ORDER BY Abbreviation").fetchall()

# Get next available PlayerID
next_pid = con.execute("SELECT COALESCE(MAX(PlayerID), 0) + 1 FROM Players").fetchone()[0]

inserted  = 0
skipped   = 0

print("Scanning NHL API rosters for missing players...\n")

for team_id, team_name, abbrev in teams:
    url = f"https://api-web.nhle.com/v1/roster/{abbrev}/current"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"  [HTTP {resp.status_code}] {team_name}")
            continue
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] {team_name}: {e}")
        continue

    all_players = (
        data.get("forwards",   []) +
        data.get("defensemen", []) +
        data.get("goalies",    [])
    )

    team_added = 0
    for p in all_players:
        first_raw = (p.get("firstName",  {}) or {}).get("default", "") or ""
        last_raw  = (p.get("lastName",   {}) or {}).get("default", "") or ""
        if not first_raw and not last_raw:
            continue

        # ASCII-normalize names for storage (no accents)
        first_store = normalize(first_raw).title()
        last_store  = normalize(last_raw).title()
        norm_full   = normalize(f"{first_raw} {last_raw}")
        norm_last   = normalize(last_raw)

        # Check if already in DB
        if by_full_team.get((norm_full, abbrev)) or by_last_team.get((norm_last, abbrev)):
            skipped += 1
            continue

        # Truly missing — insert
        pos    = safe_str(p.get("positionCode"))
        jersey = safe_int(p.get("sweaterNumber"))
        height = safe_int(p.get("heightInInches"))
        weight = safe_int(p.get("weightInPounds"))
        dob    = safe_str(p.get("birthDate"))
        city   = safe_str((p.get("birthCity",   {}) or {}).get("default"))
        country= safe_str(p.get("birthCountry"))

        con.execute("""
            INSERT INTO Players
            (PlayerID, TeamID, FirstName, LastName, Position,
             JerseyNumber, IsActive, HeightInches, WeightLbs,
             DateOfBirth, BirthCity, BirthCountry)
            VALUES (?, ?, ?, ?, ?, ?, TRUE, ?, ?, ?, ?, ?)
        """, [next_pid, team_id, first_store, last_store, pos,
              jersey, height, weight, dob, city, country])

        print(f"  [ADD] {abbrev} #{jersey} {pos} {first_store} {last_store} (PlayerID={next_pid})")

        # Update lookups so duplicates aren't inserted if player appears on multiple teams
        by_full_team[(norm_full, abbrev)] = next_pid
        by_last_team[(norm_last, abbrev)] = next_pid
        next_pid += 1
        inserted  += 1
        team_added += 1

    if team_added > 0:
        print(f"  {team_name}: added {team_added}")

    time.sleep(0.3)

print(f"\n{'='*55}")
print(f"  Players inserted: {inserted}")
print(f"  Already in DB:    {skipped}")
print(f"  New max PlayerID: {next_pid - 1}")
print(f"{'='*55}")

con.close()
print("\nDone!")
