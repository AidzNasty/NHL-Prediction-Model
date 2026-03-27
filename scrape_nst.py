"""
scrape_nst.py
--------------
Scrapes team stats, player stats, and goalie stats from Natural Stat Trick.

Sources:
    Team stats:    https://www.naturalstattrick.com/teamtable.php
    Player std:    https://www.naturalstattrick.com/playerteams.php?stdoi=std
    Player on-ice: https://www.naturalstattrick.com/playerteams.php?stdoi=oi
    Goalie stats:  https://www.naturalstattrick.com/playerteams.php?stdoi=g

USAGE:
------
python scrape_nst.py --season 2025-26
python scrape_nst.py --season 2024-25
python scrape_nst.py --season all
"""

import os
import sys
import time
import argparse
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
    "Referer": "https://www.naturalstattrick.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

SEASON_PARAMS = {
    "2025-26": {"fromseason": "20252026", "thruseason": "20252026"},
    "2024-25": {"fromseason": "20242025", "thruseason": "20242025"},
}

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
    "St Louis Blues":         "St. Louis Blues",
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

# -- Helpers --------------------------------------------------
def safe_int(val):
    try:
        return int(str(val).replace(",", "")) if val and str(val).strip() not in ("", "-") else None
    except:
        return None

def safe_float(val):
    try:
        return float(val) if val and str(val).strip() not in ("", "-") else None
    except:
        return None

def safe_str(val):
    return str(val).strip() if val else None

def fetch_table(url, table_id, params=None):
    print(f"  Fetching: {url}" + (f" params={params}" if params else ""))
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    if resp.status_code == 429:
        print("  [429] Rate limited — sleeping 30s...")
        time.sleep(30)
        resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    soup  = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": table_id})
    if not table:
        raise RuntimeError(f"Table '{table_id}' not found")
    thead   = table.find("thead")
    headers = [th.get_text(strip=True) for th in thead.find_all("th")] if thead else []
    tbody   = table.find("tbody") or table
    rows    = []
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) >= 4:
            rows.append(cells)
    print(f"  Found {len(rows)} rows, {len(headers)} columns")
    return headers, rows

def build_lookups(con):
    team_rows   = con.execute("SELECT TeamID, TeamName FROM Teams").fetchall()
    team_lookup = {r[1]: r[0] for r in team_rows}
    player_rows = con.execute("""
        SELECT p.PlayerID, p.FirstName || ' ' || p.LastName AS FullName, t.Abbreviation
        FROM Players p JOIN Teams t ON p.TeamID = t.TeamID
        WHERE p.IsActive = TRUE
    """).fetchall()
    by_name_team = {(r[1].lower(), r[2].upper()): r[0] for r in player_rows}
    by_name = {}
    for r in player_rows:
        if r[1].lower() not in by_name:
            by_name[r[1].lower()] = r[0]
    return team_lookup, by_name_team, by_name

# -- Scrape Teams ---------------------------------------------
def scrape_teams(season, con, params):
    print(f"\n{'='*55}")
    print(f"Scraping team stats — {season}")
    print(f"{'='*55}")
    _, rows = fetch_table("https://www.naturalstattrick.com/teamtable.php", "teams", params)
    team_lookup, _, _ = build_lookups(con)
    cat_id = con.execute("SELECT CategoryID FROM StatCategories WHERE CategoryName='Team Stats'").fetchone()
    cat_id = cat_id[0] if cat_id else 1
    con.execute("DELETE FROM TeamStandings WHERE Season = ?", [season])
    stat_id  = con.execute("SELECT COALESCE(MAX(StandingID), 0) FROM TeamStandings").fetchone()[0] + 1
    inserted = 0
    no_match = 0
    for row in rows:
        if len(row) < 72:
            continue
        team_name = TEAM_NAME_MAP.get(row[1], row[1])
        team_id   = team_lookup.get(team_name)
        if not team_id:
            print(f"  [NO MATCH] {row[1]}")
            no_match += 1
            continue
        try:
            vals = (
                stat_id, team_id, season,
                safe_int(row[2]), safe_str(row[3]),
                safe_int(row[4]), safe_int(row[5]), safe_int(row[6]),
                safe_int(row[7]), safe_int(row[8]), safe_float(row[9]),
                safe_int(row[10]), safe_int(row[11]), safe_float(row[12]),
                safe_int(row[13]), safe_int(row[14]), safe_float(row[15]),
                safe_int(row[16]), safe_int(row[17]), safe_float(row[18]),
                safe_int(row[19]), safe_int(row[20]), safe_float(row[21]),
                safe_float(row[22]), safe_float(row[23]), safe_float(row[24]),
                safe_int(row[25]), safe_int(row[26]), safe_float(row[27]),
                safe_int(row[28]), safe_int(row[29]), safe_float(row[30]),
                safe_int(row[31]), safe_int(row[32]), safe_float(row[33]),
                safe_float(row[34]), safe_float(row[35]),
                safe_int(row[36]), safe_int(row[37]), safe_float(row[38]),
                safe_int(row[39]), safe_int(row[40]), safe_float(row[41]),
                safe_int(row[42]), safe_int(row[43]), safe_float(row[44]),
                safe_float(row[45]), safe_float(row[46]),
                safe_int(row[47]), safe_int(row[48]), safe_float(row[49]),
                safe_int(row[50]), safe_int(row[51]), safe_float(row[52]),
                safe_int(row[53]), safe_int(row[54]), safe_float(row[55]),
                safe_float(row[56]), safe_float(row[57]),
                safe_int(row[58]), safe_int(row[59]), safe_float(row[60]),
                safe_int(row[61]), safe_int(row[62]), safe_float(row[63]),
                safe_int(row[64]), safe_int(row[65]), safe_float(row[66]),
                safe_float(row[67]), safe_float(row[68]),
                safe_float(row[69]), safe_float(row[70]), safe_float(row[71]),
            )
            placeholders = ",".join(["?"] * len(vals))
            con.execute(f"INSERT INTO TeamStandings VALUES ({placeholders})", vals)
            stat_id  += 1
            inserted += 1
        except Exception as e:
            print(f"  [ERROR] {team_name}: {e}")
    print(f"\n  Teams inserted: {inserted} | No match: {no_match}")

# -- Scrape Players --------------------------------------------
def scrape_players(season, con, params):
    print(f"\n{'='*55}")
    print(f"Scraping player stats — {season}")
    print(f"{'='*55}")
    _, std_rows = fetch_table(
        "https://www.naturalstattrick.com/playerteams.php",
        "indreg", {**params, "stdoi": "std"}
    )
    time.sleep(4)
    _, oi_rows = fetch_table(
        "https://www.naturalstattrick.com/playerteams.php",
        "players", {**params, "stdoi": "oi"}
    )
    oi_lookup = {}
    for row in oi_rows:
        if len(row) >= 6:
            key = (row[1].lower(), row[2].upper())
            oi_lookup[key] = row
    _, by_name_team, by_name = build_lookups(con)
    cat_id = con.execute("SELECT CategoryID FROM StatCategories WHERE CategoryName='Skater Stats'").fetchone()
    cat_id = cat_id[0] if cat_id else 2
    con.execute("DELETE FROM PlayerStats WHERE Season = ?", [season])
    stat_id  = con.execute("SELECT COALESCE(MAX(PlayerStatID), 0) FROM PlayerStats").fetchone()[0] + 1
    inserted = 0
    no_match = 0
    for row in std_rows:
        if len(row) < 35:
            continue
        name = row[1]
        team = row[2].upper()
        pid  = by_name_team.get((name.lower(), team)) or by_name.get(name.lower())
        if not pid:
            no_match += 1
            continue
        oi = oi_lookup.get((name.lower(), team)) or []
        def g(lst, i):
            return lst[i] if len(lst) > i else None
        try:
            vals = (
                stat_id, pid, cat_id, season,
                # Basic
                safe_int(row[4]), safe_float(row[5]),
                # Individual std (29 values)
                safe_int(row[6]),    safe_int(row[7]),    safe_int(row[8]),
                safe_int(row[9]),    safe_int(row[10]),   safe_float(row[11]),
                safe_int(row[12]),   safe_float(row[13]), safe_float(row[14]),
                safe_int(row[15]),   safe_int(row[16]),   safe_int(row[17]),
                safe_int(row[18]),   safe_int(row[19]),   safe_int(row[20]),
                safe_int(row[21]),   safe_int(row[22]),   safe_int(row[23]),
                safe_int(row[24]),   safe_int(row[25]),   safe_int(row[26]),
                safe_int(row[27]),   safe_int(row[28]),   safe_int(row[29]),
                safe_int(row[30]),   safe_int(row[31]),   safe_int(row[32]),
                safe_int(row[33]),   safe_float(row[34]),
                # On-ice (48 values)
                safe_int(g(oi,6)),   safe_int(g(oi,7)),   safe_float(g(oi,8)),
                safe_int(g(oi,9)),   safe_int(g(oi,10)),  safe_float(g(oi,11)),
                safe_int(g(oi,12)),  safe_int(g(oi,13)),  safe_float(g(oi,14)),
                safe_int(g(oi,15)),  safe_int(g(oi,16)),  safe_float(g(oi,17)),
                safe_float(g(oi,18)),safe_float(g(oi,19)),safe_float(g(oi,20)),
                safe_int(g(oi,21)),  safe_int(g(oi,22)),  safe_float(g(oi,23)),
                safe_int(g(oi,24)),  safe_int(g(oi,25)),  safe_float(g(oi,26)),
                safe_int(g(oi,27)),  safe_int(g(oi,28)),  safe_float(g(oi,29)),
                safe_int(g(oi,30)),  safe_int(g(oi,31)),  safe_float(g(oi,32)),
                safe_int(g(oi,33)),  safe_int(g(oi,34)),  safe_float(g(oi,35)),
                safe_int(g(oi,36)),  safe_int(g(oi,37)),  safe_float(g(oi,38)),
                safe_int(g(oi,39)),  safe_int(g(oi,40)),  safe_float(g(oi,41)),
                safe_float(g(oi,42)),safe_float(g(oi,43)),safe_float(g(oi,44)),
                safe_int(g(oi,45)),  safe_int(g(oi,46)),  safe_int(g(oi,47)),
                safe_int(g(oi,48)),  safe_float(g(oi,49)),
                safe_int(g(oi,50)),  safe_int(g(oi,51)),  safe_int(g(oi,52)),
                safe_float(g(oi,53)),
            )
            placeholders = ",".join(["?"] * len(vals))
            con.execute(f"INSERT INTO PlayerStats VALUES ({placeholders})", vals)
            stat_id  += 1
            inserted += 1
        except Exception as e:
            print(f"  [ERROR] {name} ({team}): {e}")
    print(f"\n  Players inserted: {inserted} | No match: {no_match}")

# -- Scrape Goalies --------------------------------------------
def scrape_goalies(season, con, params):
    print(f"\n{'='*55}")
    print(f"Scraping goalie stats — {season}")
    print(f"{'='*55}")
    _, rows = fetch_table(
        "https://www.naturalstattrick.com/playerteams.php",
        "players", {**params, "stdoi": "g"}
    )
    _, by_name_team, by_name = build_lookups(con)
    con.execute("DELETE FROM GoalieStats WHERE Season = ?", [season])
    stat_id  = con.execute("SELECT COALESCE(MAX(GoalieStatID), 0) FROM GoalieStats").fetchone()[0] + 1
    inserted = 0
    no_match = 0
    for row in rows:
        if len(row) < 34:
            continue
        name = row[1]
        team = row[2].upper()
        pid  = by_name_team.get((name.lower(), team)) or by_name.get(name.lower())
        if not pid:
            no_match += 1
            continue
        xga    = safe_float(row[11])
        ga     = safe_int(row[7])
        toi    = safe_float(row[4])
        gsax   = round(xga - ga, 2) if (xga is not None and ga is not None) else None
        gsax60 = round(gsax / toi * 60, 3) if (gsax is not None and toi and toi > 0) else None
        try:
            vals = (
                stat_id, pid, season,
                safe_int(row[3]),    safe_float(row[4]),
                safe_int(row[5]),    safe_int(row[6]),    ga,
                safe_float(row[8]),  safe_float(row[9]),  safe_float(row[10]), xga,
                safe_int(row[12]),   safe_int(row[13]),   safe_int(row[14]),
                safe_float(row[15]), safe_float(row[16]), safe_float(row[17]),
                safe_int(row[18]),   safe_int(row[19]),   safe_int(row[20]),
                safe_float(row[21]), safe_float(row[22]), safe_float(row[23]),
                safe_int(row[24]),   safe_int(row[25]),   safe_int(row[26]),
                safe_float(row[27]), safe_float(row[28]), safe_float(row[29]),
                safe_int(row[30]),   safe_int(row[31]),
                safe_float(row[32]), safe_float(row[33]),
                gsax, gsax60,
            )
            placeholders = ",".join(["?"] * len(vals))
            con.execute(f"INSERT INTO GoalieStats VALUES ({placeholders})", vals)
            stat_id  += 1
            inserted += 1
        except Exception as e:
            print(f"  [ERROR] {name} ({team}): {e}")
    print(f"\n  Goalies inserted: {inserted} | No match: {no_match}")

# -- Main -----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--season", required=True, help="Season: 2025-26, 2024-25, or 'all'")
args = parser.parse_args()

seasons = list(SEASON_PARAMS.keys()) if args.season == "all" else [args.season]
if args.season != "all" and args.season not in SEASON_PARAMS:
    print(f"Unknown season: {args.season}")
    sys.exit(1)

print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

for season in seasons:
    params = SEASON_PARAMS[season]
    scrape_teams(season, con, params)
    time.sleep(4)
    scrape_players(season, con, params)
    time.sleep(4)
    scrape_goalies(season, con, params)
    time.sleep(4)

ts = con.execute("SELECT COUNT(*) FROM TeamStandings").fetchone()[0]
ps = con.execute("SELECT COUNT(*) FROM PlayerStats").fetchone()[0]
gs = con.execute("SELECT COUNT(*) FROM GoalieStats").fetchone()[0]
print(f"\n{'='*55}")
print(f"  TeamStandings: {ts} rows")
print(f"  PlayerStats:   {ps} rows")
print(f"  GoalieStats:   {gs} rows")
print(f"{'='*55}")
con.close()
print("\nDone!")
