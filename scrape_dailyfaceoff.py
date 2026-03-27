"""
scrape_dailyfaceoff.py
-----------------------
Scrapes Daily Faceoff player news and writes to PlayerGameStatus in MotherDuck.

Classifies each news item into a LineupStatus:
    Active          - confirmed playing tonight (line change in, goalie start, returning from injury)
    Injured         - confirmed out (injury, illness)
    Questionable    - day-to-day or no timeline given
    Scratched       - healthy scratch
    Traded          - traded away
    Roster Move     - called up / sent down / waiver

Maps each item to a player in the Players table by first+last name fuzzy match,
and to a game in the Games table by team + today's date.

USAGE:
------
# Run manually:
python scrape_dailyfaceoff.py

# Scheduled (see schedule_daily.py):
# - Morning run  ~9:00 AM  (early injury/lineup news)
# - Pre-game run ~1 hr before each puck drop
"""

import os
import re
import time
import duckdb
import requests
from datetime import date, datetime, timezone
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


load_dotenv()

TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

if not TOKEN:
    raise ValueError("MOTHERDUCK_TOKEN not found in .env file")

URL = "https://www.dailyfaceoff.com/hockey-player-news"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# -- Team name normalization ----------------------------------
# Maps DailyFaceoff team names -> our Teams table names
TEAM_NAME_MAP = {
    "Anaheim Ducks":          "Anaheim Ducks",
    "Boston Bruins":          "Boston Bruins",
    "Buffalo Sabres":         "Buffalo Sabres",
    "Calgary Flames":         "Calgary Flames",
    "Carolina Hurricanes":    "Carolina Hurricanes",
    "Chicago Blackhawks":     "Chicago Blackhawks",
    "Colorado Avalanche":     "Colorado Avalanche",
    "Columbus Blue Jackets":  "Columbus Blue Jackets",
    "Dallas Stars":           "Dallas Stars",
    "Detroit Red Wings":      "Detroit Red Wings",
    "Edmonton Oilers":        "Edmonton Oilers",
    "Florida Panthers":       "Florida Panthers",
    "Los Angeles Kings":      "Los Angeles Kings",
    "Minnesota Wild":         "Minnesota Wild",
    "Montreal Canadiens":     "Montreal Canadiens",
    "Nashville Predators":    "Nashville Predators",
    "New Jersey Devils":      "New Jersey Devils",
    "New York Islanders":     "New York Islanders",
    "New York Rangers":       "New York Rangers",
    "Ottawa Senators":        "Ottawa Senators",
    "Philadelphia Flyers":    "Philadelphia Flyers",
    "Pittsburgh Penguins":    "Pittsburgh Penguins",
    "San Jose Sharks":        "San Jose Sharks",
    "Seattle Kraken":         "Seattle Kraken",
    "St. Louis Blues":        "St. Louis Blues",
    "Tampa Bay Lightning":    "Tampa Bay Lightning",
    "Toronto Maple Leafs":    "Toronto Maple Leafs",
    "Utah Hockey Club":       "Utah Mammoth",
    "Utah Mammoth":           "Utah Mammoth",
    "Vancouver Canucks":      "Vancouver Canucks",
    "Vegas Golden Knights":   "Vegas Golden Knights",
    "Washington Capitals":    "Washington Capitals",
    "Winnipeg Jets":          "Winnipeg Jets",
}

# -- LineupStatus classification ------------------------------
def classify_status(category: str, blurb: str) -> tuple[str, str]:
    """
    Returns (LineupStatus, InjuryDetail) based on news category and blurb text.
    LineupStatus values: Active, Injured, Questionable, Scratched, Traded, Roster Move
    """
    cat   = category.lower()
    text  = blurb.lower()

    # Goalie start -> Starting Goalie
    if "goalie start" in cat or "goalie start" in text:
        return "Starting Goalie", None

    # Line change — check if drawing IN or OUT
    if "line change" in cat or "line change" in text:
        if any(w in text for w in ["draw back", "draws back", "draw in", "draws in",
                                    "return to", "returns to", "re-enter", "re-enters",
                                    "back in", "slot back", "will play"]):
            return "Active", None
        if any(w in text for w in ["out of", "scratch", "sit out", "won't play",
                                    "will not play", "miss", "removed"]):
            return "Scratched", None
        return "Active", None  # default line change = in lineup

    # Injury
    if "injury" in cat or "injured" in cat:
        # Extract injury detail from blurb (e.g. "upper-body", "lower-body", "knee")
        injury_detail = _extract_injury(blurb)

        if any(w in text for w in ["return", "returns", "back in", "cleared",
                                    "will play", "no longer", "removed from"]):
            return "Active", injury_detail

        if any(w in text for w in ["day-to-day", "dtd", "no timeline", "week-to-week",
                                    "questionable", "game-time decision", "gtd"]):
            return "Questionable", injury_detail

        return "Injured", injury_detail

    # Roster moves
    if any(w in cat for w in ["roster", "waiver", "recall", "reassign"]):
        return "Roster Move", None

    # Trade
    if "trade" in cat or "traded" in text:
        return "Traded", None

    # Signing
    if "sign" in cat:
        return "Active", None

    return "Active", None


def _extract_injury(text: str) -> str | None:
    """Pull injury body part from blurb text."""
    patterns = [
        r'\((upper[- ]body)\)', r'\((lower[- ]body)\)', r'\((head)\)',
        r'\((knee)\)', r'\((shoulder)\)', r'\((back)\)', r'\((hip)\)',
        r'\((ankle)\)', r'\((wrist)\)', r'\((hand)\)', r'\((foot)\)',
        r'\((groin)\)', r'\((concussion)\)', r'\((illness)\)',
        r'\((undisclosed)\)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).title()
    # Generic fallback — grab first parenthetical
    m = re.search(r'\(([^)]{3,30})\)', text)
    if m:
        val = m.group(1)
        if not any(c.isdigit() for c in val):
            return val.title()
    return None


# -- Scraper --------------------------------------------------
def scrape_news() -> list[dict]:
    """Fetch and parse all news items from Daily Faceoff."""
    print(f"Fetching {URL}...")
    resp = requests.get(URL, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from Daily Faceoff")

    soup  = BeautifulSoup(resp.text, "html.parser")
    items = []

    # Each news item is a container with player name, team, category, blurb
    # Structure: player link -> team link -> category text -> blurb paragraph
    # We find player news blocks by looking for the player name links
    for player_link in soup.find_all("a", href=re.compile(r"/players/news/")):
        try:
            # Player full name is in the link text, strip position suffix
            raw_name = player_link.get_text(strip=True)
            # Name format: "David Pastrnak(Right Wing)(RW)" -> split on "("
            player_name = raw_name.split("(")[0].strip()
            if not player_name or len(player_name) < 3:
                continue

            # Walk up to the containing block to find team + category + blurb
            container = player_link.find_parent()
            for _ in range(6):
                if container is None:
                    break
                container = container.find_parent()
                if container and container.find("a", href=re.compile(r"/teams/")):
                    break

            if container is None:
                continue

            # Team name
            team_link = container.find("a", href=re.compile(r"/teams/"))
            team_name = team_link.get_text(strip=True) if team_link else None
            if not team_name:
                continue

            # Category (Injury, Line Change, Goalie Start, etc.)
            all_text = container.get_text(separator="|", strip=True)
            category = "Unknown"
            for cat in ["Injury", "Line Change", "Goalie Start", "Signing",
                        "Trade", "Roster Move", "Waiver Move"]:
                if cat.lower() in all_text.lower():
                    category = cat
                    break

            # Blurb — the main paragraph text
            paragraphs = container.find_all("p")
            blurb = " ".join(p.get_text(strip=True) for p in paragraphs[:2]) if paragraphs else ""

            # Timestamp from source link
            timestamp = None
            for a in container.find_all("a", href=re.compile(r"x\.com|twitter\.com")):
                ts_text = a.get_text(strip=True)
                m = re.search(r'(\d{4}-\d{2}-\d{2}T[\d:\.Z]+)', ts_text)
                if m:
                    try:
                        timestamp = datetime.fromisoformat(m.group(1).replace("Z",""))
                    except:
                        pass

            items.append({
                "player_name": player_name,
                "team_name":   TEAM_NAME_MAP.get(team_name, team_name),
                "category":    category,
                "blurb":       blurb[:500],  # cap at 500 chars
                "timestamp":   timestamp or datetime.now(timezone.utc),
            })

        except Exception as e:
            continue

    # Deduplicate by player name (keep most recent)
    seen = {}
    for item in items:
        key = item["player_name"]
        if key not in seen:
            seen[key] = item
    return list(seen.values())


# -- Database operations ---------------------------------------
def load_to_db(items: list[dict], con: duckdb.DuckDBPyConnection):
    today = str(date.today())

    # Load player and game lookups
    players = con.execute("""
        SELECT p.PlayerID, p.FirstName || ' ' || p.LastName AS FullName, t.TeamName
        FROM Players p
        JOIN Teams t ON p.TeamID = t.TeamID
        WHERE p.IsActive = TRUE
    """).fetchall()
    player_lookup = {(row[1].lower(), row[2]): row[0] for row in players}
    player_name_only = {row[1].lower(): row[0] for row in players}

    games = con.execute(f"""
        SELECT g.GameID, t1.TeamName AS HomeTeam, t2.TeamName AS AwayTeam
        FROM Games g
        JOIN Teams t1 ON g.HomeTeamID = t1.TeamID
        JOIN Teams t2 ON g.AwayTeamID = t2.TeamID
        WHERE g.GameDate = '{today}'
    """).fetchall()

    # Build team -> gameID map (team appears in game as home OR away)
    team_to_game = {}
    for game_id, home, away in games:
        team_to_game[home] = game_id
        team_to_game[away] = game_id

    if not games:
        print(f"  No games found in DB for {today} — status updates will have NULL GameID")

    inserted = 0
    skipped  = 0
    no_match = 0

    for item in items:
        name      = item["player_name"]
        team      = item["team_name"]
        category  = item["category"]
        blurb     = item["blurb"]
        timestamp = item["timestamp"]

        # Resolve PlayerID
        player_id = (
            player_lookup.get((name.lower(), team)) or
            player_name_only.get(name.lower())
        )
        if not player_id:
            print(f"  [NO MATCH] {name} ({team})")
            no_match += 1
            continue

        # Resolve GameID (NULL if player's team not playing today)
        game_id = team_to_game.get(team)

        # Classify status
        status, injury_detail = classify_status(category, blurb)

        # Skip if player's team isn't playing today — GameID required (NOT NULL)
        if not game_id:
            continue

        # Check if already exists for this player+game today (upsert logic)
        existing = con.execute("""
            SELECT StatusID FROM PlayerGameStatus
            WHERE PlayerID = ? AND GameID = ?
        """, [player_id, game_id]).fetchone()

        # Get next StatusID for inserts
        next_id = con.execute("SELECT COALESCE(MAX(StatusID), 0) + 1 FROM PlayerGameStatus").fetchone()[0]

        if existing:
            # Update existing row
            con.execute("""
                UPDATE PlayerGameStatus
                SET Status = ?, StatusDate = ?
                WHERE StatusID = ?
            """, [status, timestamp, existing[0]])
            print(f"  [UPDATE] {name} ({team}) -> {status}" + (f" [{injury_detail}]" if injury_detail else ""))
        else:
            # Insert new row
            con.execute("""
                INSERT INTO PlayerGameStatus (StatusID, PlayerID, GameID, Status, StatusDate)
                VALUES (?, ?, ?, ?, ?)
            """, [next_id, player_id, game_id, status, timestamp])
            inserted += 1
            print(f"  [INSERT] {name} ({team}) -> {status}" + (f" [{injury_detail}]" if injury_detail else ""))

    return inserted, skipped, no_match


# -- Main -----------------------------------------------------
def main():
    print("=" * 55)
    print(f"Daily Faceoff Scraper — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Scrape
    items = scrape_news()
    print(f"\nNews items found: {len(items)}\n")

    if not items:
        print("No items scraped — check if Daily Faceoff changed their HTML structure.")
        return

    # Connect and load
    print(f"Connecting to MotherDuck: {DB}...")
    con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
    print("Connected!\n")

    inserted, skipped, no_match = load_to_db(items, con)

    # Summary
    total = con.execute("SELECT COUNT(*) FROM PlayerGameStatus").fetchone()[0]
    today_count = con.execute(f"""
        SELECT COUNT(*) FROM PlayerGameStatus
        WHERE CAST(StatusDate AS DATE) = '{date.today()}'
    """).fetchone()[0]

    print(f"\n{'='*55}")
    print(f"  Items processed:   {len(items)}")
    print(f"  Inserted:          {inserted}")
    print(f"  Updated:           {len(items) - inserted - no_match}")
    print(f"  No player match:   {no_match}")
    print(f"  Today's records:   {today_count}")
    print(f"  Total in DB:       {total}")
    print(f"{'='*55}")
    con.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
