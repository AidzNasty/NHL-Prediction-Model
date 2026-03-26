"""
scrape_nhl_lineups.py
----------------------
Replaces scrape_dailyfaceoff.py entirely.

Uses NHL gamecenter API to get:
  - Starting goalies (starter = True)
  - Active players (dressed for tonight)
  - Scratches (on roster but not in playerByGameStats)

For games in PRE state:  gets projected lineup
For games in LIVE/OFF:   gets confirmed lineup + actual starter

Updates PlayerGameStatus table in MotherDuck.

Run at 9:00 AM (morning skate), 6:00 PM (confirmed lines).

Usage: python scrape_nhl_lineups.py
"""

import os
import time
import requests
import duckdb
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

NHL_API_ABBREVS = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF", "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL", "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL", "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD", "New York Islanders": "NYI",
    "New York Rangers": "NYR", "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR", "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}

def safe_str(val):
    return str(val).strip() if val else None

print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

# ── Build player lookups ──────────────────────────────────────
# NHL API uses their own player IDs — we need to map to our PlayerIDs
# Try matching by NHL API ID first, then fall back to name matching
print("Building player lookups...")

players = con.execute("""
    SELECT p.PlayerID, p.FirstName, p.LastName,
           p.FirstName || ' ' || p.LastName AS FullName,
           t.Abbreviation
    FROM Players p
    JOIN Teams t ON p.TeamID = t.TeamID
    WHERE p.IsActive = TRUE
""").fetchall()

# Name-based lookup (normalized)
import unicodedata
def normalize(s):
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd
                   if unicodedata.category(c) != "Mn").lower().strip()

by_name = {}
by_name_team = {}
for pid, first, last, full, abbrev in players:
    key = normalize(full)
    if key not in by_name:
        by_name[key] = pid
    by_name_team[(key, abbrev.upper())] = pid

print(f"  {len(players)} active players loaded")

# ── Get today's games ─────────────────────────────────────────
today = date.today()
print(f"\nFetching schedule for {today}...")

schedule_url = f"https://api-web.nhle.com/v1/schedule/{today}"
resp = requests.get(schedule_url, headers=HEADERS, timeout=10)
if resp.status_code != 200:
    print(f"  ERROR: {resp.status_code}")
    con.close()
    exit()

schedule = resp.json()
games_today = []
for week in schedule.get("gameWeek", []):
    if str(week.get("date", ""))[:10] == str(today):
        for game in week.get("games", []):
            games_today.append(game)

print(f"  {len(games_today)} games today")

# ── Process each game ─────────────────────────────────────────
total_updated  = 0
total_starters = 0

for game in games_today:
    gid        = game["id"]
    state      = game.get("gameState", "")
    home_abbrev = game.get("homeTeam", {}).get("abbrev", "")
    away_abbrev = game.get("awayTeam", {}).get("abbrev", "")

    print(f"\n  {away_abbrev} @ {home_abbrev} — State: {state} (ID: {gid})")

    # Get boxscore for this game
    box_url = f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore"
    box_resp = requests.get(box_url, headers=HEADERS, timeout=10)

    if box_resp.status_code != 200:
        print(f"    Boxscore not available yet ({box_resp.status_code})")
        continue

    box = box_resp.json()
    pbg = box.get("playerByGameStats", {})

    if not pbg:
        print(f"    No playerByGameStats yet")
        continue

    status_date = datetime.now()

    for side, abbrev in [("homeTeam", home_abbrev), ("awayTeam", away_abbrev)]:
        side_data = pbg.get(side, {})
        if not side_data:
            continue

        # Map NHL API abbrev to DB abbrev
        abbrev_map = {
            "NJD": "N.J", "SJS": "S.J", "TBL": "T.B",
            "LAK": "L.A", "VGK": "VGK", "UTA": "UTA",
        }
        db_abbrev = abbrev_map.get(abbrev, abbrev)

        team_row = con.execute(f"""
            SELECT TeamID FROM Teams
            WHERE Abbreviation = '{db_abbrev}'
        """).fetchone()

        if not team_row:
            team_name = game.get(side, {}).get("name", {}).get("default", "")
            team_row = con.execute(f"""
                SELECT TeamID FROM Teams WHERE TeamName = ?
            """, [team_name]).fetchone()

        if not team_row:
            print(f"    [NO TEAM] {abbrev} / {db_abbrev}")
            continue

        team_id = team_row[0]

        # Get all dressed players
        dressed_ids = set()
        all_groups  = (
            [(p, "Active") for p in side_data.get("forwards", [])] +
            [(p, "Active") for p in side_data.get("defense",  [])] +
            [(p, "Active") for p in side_data.get("goalies",  [])]
        )

        for p_data, status in all_groups:
            api_name = p_data.get("name", {}).get("default", "")
            # API name format is "F. LastName" — expand to full
            # Try to get full name from sweater number lookup
            sweater = p_data.get("sweaterNumber")
            nhl_pid = p_data.get("playerId")

            # Normalize the abbreviated name for matching
            # "J. Swayman" → try to find Swayman on this team
            name_parts = api_name.replace(".", "").split()
            last_name  = name_parts[-1] if name_parts else ""
            norm_last  = normalize(last_name)

            # Find in our DB by last name + team
            our_pid = None
            for (norm_full, team_ab), pid in by_name_team.items():
                if norm_last and norm_full.endswith(norm_last) and team_ab == abbrev:
                    our_pid = pid
                    break

            # Fallback: last name only
            if not our_pid:
                for norm_full, pid in by_name.items():
                    if norm_last and norm_full.endswith(norm_last):
                        our_pid = pid
                        break

            if not our_pid:
                continue

            dressed_ids.add(our_pid)

            is_starter = p_data.get("starter", False)
            player_status = "Active"
            if is_starter and p_data.get("position") == "G":
                player_status = "Starting Goalie"

            # Upsert into PlayerGameStatus
            existing = con.execute(f"""
                SELECT StatusID FROM PlayerGameStatus
                WHERE PlayerID = {our_pid}
                  AND CAST(StatusDate AS DATE) = '{today}'
            """).fetchone()

            if existing:
                con.execute(f"""
                    UPDATE PlayerGameStatus SET
                        Status     = ?,
                        StatusDate = ?
                    WHERE StatusID = {existing[0]}
                """, [player_status, status_date])
            else:
                max_id = con.execute(
                    "SELECT COALESCE(MAX(StatusID), 0) FROM PlayerGameStatus"
                ).fetchone()[0] + 1
                con.execute("""
                    INSERT INTO PlayerGameStatus
                    (StatusID, PlayerID, Status, StatusDate, Notes)
                    VALUES (?, ?, ?, ?, ?)
                """, [max_id, our_pid, player_status,
                      status_date, f"Game {gid}"])

            total_updated += 1
            if player_status == "Starting Goalie":
                total_starters += 1
                print(f"    ⭐ STARTER: {api_name} ({abbrev})")

        print(f"    {abbrev}: {len(dressed_ids)} dressed players found")

    time.sleep(0.5)

# ── Summary ───────────────────────────────────────────────────
total_statuses = con.execute(
    f"SELECT COUNT(*) FROM PlayerGameStatus WHERE CAST(StatusDate AS DATE) = '{today}'"
).fetchone()[0]

print(f"\n{'='*55}")
print(f"  Players updated:    {total_updated}")
print(f"  Starting goalies:   {total_starters}")
print(f"  Total in DB today:  {total_statuses}")
print(f"{'='*55}")

con.close()
print("\nDone!")
# DEBUG — run this separately to check schedule structure
if __name__ == "__debug__":
    import requests, json
    r = requests.get("https://api-web.nhle.com/v1/schedule/2026-03-25",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    d = r.json()
    for week in d.get("gameWeek", []):
        print(f"Week date: {week.get('date')}, games: {len(week.get('games',[]))}")
        for g in week.get("games", []):
            print(f"  {g.get('id')} — {g.get('gameDate')} — "
                  f"{g.get('awayTeam',{}).get('abbrev')} @ "
                  f"{g.get('homeTeam',{}).get('abbrev')}")
