"""
fix_mangled_names.py
---------------------
Fixes player names that were mangled during initial import
(accent characters dropped mid-word rather than substituted).

Examples:
  "Neas"   should be "Necas"   (Nečas)
  "Pastrk" should be "Pastrnák" -> stored as "Pastrnak"

Strategy:
1. Pull all players from NHL API (all 32 teams)
2. For each API player, normalize their name to ASCII
3. Try to find them in our DB by normalized last name + team
4. If the DB name looks mangled (much shorter than API name),
   update the DB name to the ASCII-normalized API version
5. Then fill in bio data

Usage: python fix_mangled_names.py
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

# DB abbrev -> API abbrev mapping for lookup
DB_TO_API = {
    "LAK": "LAK", "NJD": "NJD", "SJS": "SJS", "TBL": "TBL",
}

def normalize(s):
    """Strip ALL diacritics -> ASCII lowercase."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    for k, v in {
        "ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "Æ": "AE",
        "ð": "d", "þ": "th", "\u0142": "l", "\u0141": "L"
    }.items():
        stripped = stripped.replace(k, v)
    return stripped.lower().strip()

def is_mangled(db_name, api_name_normalized):
    """
    Returns True if the DB name looks like a mangled version
    of the API name — i.e. it's a substring or much shorter.
    e.g. db="pastrk", api_norm="pastrnak" -> mangled
         db="necas",  api_norm="necas"    -> not mangled (already fine)
    """
    db  = db_name.lower().strip()
    api = api_name_normalized.lower().strip()
    if db == api:
        return False
    # Mangled = DB name is contained in API name but shorter
    if db in api and len(db) < len(api) - 1:
        return True
    # Or API name starts with DB name (prefix truncation)
    if api.startswith(db) and len(db) < len(api) - 1:
        return True
    return False

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

# Get all active players from DB
db_players = con.execute("""
    SELECT p.PlayerID, p.FirstName, p.LastName, t.Abbreviation, t.TeamName
    FROM Players p
    JOIN Teams t ON p.TeamID = t.TeamID
    WHERE p.IsActive = TRUE
""").fetchall()

# Build lookup: (norm_last_prefix, abbrev) -> (PlayerID, db_first, db_last)
# We use prefix matching since mangled names are truncated
db_by_norm = {}
for pid, first, last, abbrev, team_name in db_players:
    norm_last = normalize(last)
    key = (norm_last, abbrev.upper())
    db_by_norm[key] = (pid, first, last)

print(f"Loaded {len(db_players)} active players from DB")
print("Fetching all NHL API rosters...\n")

name_fixes  = 0
bio_updates = 0
no_match    = 0

for team_name, api_abbrev in NHL_API_ABBREVS.items():
    try:
        resp = requests.get(
            f"https://api-web.nhle.com/v1/roster/{api_abbrev}/current",
            headers=HEADERS, timeout=10
        )
        if resp.status_code != 200:
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

    # DB Teams table uses the same abbreviations as the NHL API
    db_abbrev = api_abbrev

    for p in all_players:
        first_raw = p.get("firstName",  {}).get("default", "") or ""
        last_raw  = p.get("lastName",   {}).get("default", "") or ""

        api_first_norm = normalize(first_raw)
        api_last_norm  = normalize(last_raw)

        # Try exact normalized match first
        key = (api_last_norm, db_abbrev)
        match = db_by_norm.get(key)

        if not match:
            # Try without abbrev — scan all teams for this last name
            for (nl, ab), val in db_by_norm.items():
                if nl == api_last_norm:
                    match = val
                    break

        if not match:
            # Try prefix match — DB name is truncated version of API name
            for (nl, ab), val in db_by_norm.items():
                if ab == db_abbrev and api_last_norm.startswith(nl) and len(nl) >= 4:
                    match = val
                    break

        if not match:
            no_match += 1
            continue

        pid, db_first, db_last = match

        # Check if name needs fixing
        db_last_norm  = normalize(db_last)
        db_first_norm = normalize(db_first)

        name_fixed = False
        if is_mangled(db_last_norm, api_last_norm):
            new_last = api_last_norm.title()
            print(f"  FIXING NAME: {db_first} {db_last:15} -> {first_raw} {last_raw} (stored as {new_last})")
            con.execute("""
                UPDATE Players SET LastName = ? WHERE PlayerID = ?
            """, [new_last, pid])
            name_fixes += 1
            name_fixed = True

        if is_mangled(db_first_norm, api_first_norm):
            new_first = api_first_norm.title()
            print(f"  FIXING FIRST: {db_first:15} -> {first_raw} (stored as {new_first})")
            con.execute("""
                UPDATE Players SET FirstName = ? WHERE PlayerID = ?
            """, [new_first, pid])
            name_fixed = True

        # Fill in bio data regardless
        jersey  = safe_int(p.get("sweaterNumber"))
        height  = safe_int(p.get("heightInInches"))
        weight  = safe_int(p.get("weightInPounds"))
        dob     = safe_str(p.get("birthDate"))
        city    = safe_str(p.get("birthCity",  {}).get("default"))
        country = safe_str(p.get("birthCountry"))

        con.execute("""
            UPDATE Players SET
                JerseyNumber = COALESCE(JerseyNumber, ?),
                HeightInches = COALESCE(HeightInches, ?),
                WeightLbs    = COALESCE(WeightLbs,    ?),
                DateOfBirth  = COALESCE(DateOfBirth,  ?),
                BirthCity    = COALESCE(BirthCity,    ?),
                BirthCountry = COALESCE(BirthCountry, ?)
            WHERE PlayerID = ?
        """, [jersey, height, weight, dob, city, country, pid])
        bio_updates += 1

    time.sleep(0.3)

# Final summary
jersey_count = con.execute("SELECT COUNT(*) FROM Players WHERE JerseyNumber IS NOT NULL").fetchone()[0]
dob_count    = con.execute("SELECT COUNT(*) FROM Players WHERE DateOfBirth   IS NOT NULL").fetchone()[0]
still_null   = con.execute("SELECT COUNT(*) FROM Players WHERE JerseyNumber IS NULL AND IsActive = TRUE").fetchone()[0]

print(f"\n{'='*55}")
print(f"  Name fixes:           {name_fixes}")
print(f"  Bio updates:          {bio_updates}")
print(f"  Players w/ jersey#:   {jersey_count}")
print(f"  Players w/ DOB:       {dob_count}")
print(f"  Still missing jersey: {still_null} (AHL/minors/LTIR)")
print(f"{'='*55}")

# Show sample of fixed players
print("\nSample — checking Necas and Pastrnak:")
sample = con.execute("""
    SELECT FirstName, LastName, JerseyNumber, DateOfBirth
    FROM Players
    WHERE LastName IN ('Necas', 'Pastrnak', 'Neas', 'Pastrk')
""").fetchall()
for r in sample:
    print(f"  {r[0]:15} {r[1]:15} #{r[2]} DOB={r[3]}")

con.close()
print("\nDone!")
