"""
scrape_gamestats.py
--------------------
Scrapes team-level box score stats from Hockey Reference.

URL format: https://www.hockey-reference.com/boxscores/YYYYMMDD0AAA.html
Example:    https://www.hockey-reference.com/boxscores/202603120BOS.html

STATS COLLECTED (aggregated by summing player rows):
    From XXX_skaters table:
        [14] Shots (S)
        [6]  PenaltyMinutes (PIM)
        [8]  PowerPlayGoals (PP Goals)
    From XXX_adv_ALLAll table:
        HIT  Hits
        BLK  BlockedShots
    Not available in box score (stored as NULL):
        FaceoffWinPct, Giveaways, Takeaways

USAGE:
------
python scrape_gamestats.py --season 2025-26 --limit 5   # test
python scrape_gamestats.py --season 2025-26              # full season
python scrape_gamestats.py --season 2024-25              # backfill
python scrape_gamestats.py --date yesterday              # nightly automation
python scrape_gamestats.py --date 2026-03-12             # specific date
"""

import os
import sys
import time
import random
import argparse
import requests
import duckdb
from datetime import date, timedelta
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

# ============================================================
# Hockey Reference abbreviation map
# ============================================================
# Hockey Reference abbreviation map.
# Used for both box score URLs and in-page table IDs (e.g. VEG_skaters).
# Confirmed via check_hr_abbrevs.py — URL abbrev == table abbrev for all teams.
# Only known difference from official NHL abbreviations: VEG (not VGK)
# ============================================================
HR_ABBREV = {
    "Anaheim Ducks":          "ANA",
    "Boston Bruins":          "BOS",
    "Buffalo Sabres":         "BUF",
    "Calgary Flames":         "CGY",
    "Carolina Hurricanes":    "CAR",
    "Chicago Blackhawks":     "CHI",
    "Colorado Avalanche":     "COL",
    "Columbus Blue Jackets":  "CBJ",
    "Dallas Stars":           "DAL",
    "Detroit Red Wings":      "DET",
    "Edmonton Oilers":        "EDM",
    "Florida Panthers":       "FLA",
    "Los Angeles Kings":      "LAK",
    "Minnesota Wild":         "MIN",
    "Montreal Canadiens":     "MTL",
    "Nashville Predators":    "NSH",
    "New Jersey Devils":      "NJD",
    "New York Islanders":     "NYI",
    "New York Rangers":       "NYR",
    "Ottawa Senators":        "OTT",
    "Philadelphia Flyers":    "PHI",
    "Pittsburgh Penguins":    "PIT",
    "San Jose Sharks":        "SJS",
    "Seattle Kraken":         "SEA",
    "St. Louis Blues":        "STL",
    "Tampa Bay Lightning":    "TBL",
    "Toronto Maple Leafs":    "TOR",
    "Utah Mammoth":           "UTA",
    "Vancouver Canucks":      "VAN",
    "Vegas Golden Knights":   "VEG",  # HR uses VEG, not VGK
    "Washington Capitals":    "WSH",
    "Winnipeg Jets":          "WPG",
}

# ============================================================
# Parse arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--season", help="Filter by season e.g. 2025-26", default=None)
parser.add_argument("--date",   help="Specific date YYYY-MM-DD or 'yesterday'", default=None)
parser.add_argument("--limit",  help="Max games to scrape (for testing)", type=int, default=None)
args = parser.parse_args()

# ============================================================
# Connect to MotherDuck
# ============================================================
print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

# ============================================================
# Build team lookups
# ============================================================
team_rows         = con.execute("SELECT TeamID, TeamName FROM Teams").fetchall()
team_id_to_name   = {r[0]: r[1] for r in team_rows}
team_id_to_abbrev = {r[0]: HR_ABBREV.get(r[1], "") for r in team_rows}

# ============================================================
# Build game query
# ============================================================
where_clauses = [
    "gs.GameID IS NULL",
    f"g.GameDate <= '{date.today()}'"
]

if args.date:
    target = str(date.today() - timedelta(days=1)) if args.date == "yesterday" else args.date
    where_clauses.append(f"g.GameDate = '{target}'")
    print(f"Mode: single date — {target}")
elif args.season:
    where_clauses.append(f"g.Season = '{args.season}'")
    print(f"Mode: full season — {args.season}")
else:
    print("Mode: all games without GameStats")

where_sql = " AND ".join(where_clauses)
query = f"""
    SELECT g.GameID, g.GameDate, g.HomeTeamID, g.AwayTeamID
    FROM Games g
    LEFT JOIN GameStats gs ON g.GameID = gs.GameID
    WHERE {where_sql}
    GROUP BY g.GameID, g.GameDate, g.HomeTeamID, g.AwayTeamID
    ORDER BY g.GameDate ASC
"""

games = con.execute(query).fetchall()
if args.limit:
    games = games[:args.limit]

print(f"Games to scrape: {len(games)}\n")

if len(games) == 0:
    print("Nothing to scrape — all games already have GameStats.")
    con.close()
    sys.exit(0)

# ============================================================
# Scraping helpers
# ============================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def build_url(game_date, home_team_id):
    date_str = str(game_date)[:10].replace("-", "")
    abbrev   = team_id_to_abbrev.get(home_team_id, "")
    return f"https://www.hockey-reference.com/boxscores/{date_str}0{abbrev}.html"

def sum_col_by_stat(table, stat_name):
    """Sum a column by data-stat attribute — robust to column order changes."""
    if table is None:
        return None
    total = 0
    found = False
    for row in table.find("tbody").find_all("tr"):
        cell = row.find(["td","th"], {"data-stat": stat_name})
        if cell:
            found = True
            try:
                total += int(cell.get_text(strip=True))
            except:
                pass
    return total if found else None

def scrape_box_score(url, home_abbrev, away_abbrev):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)

        if resp.status_code == 404:
            print(f"  [404] Not found: {url}")
            return None
        if resp.status_code == 429:
            print(f"  [429] Rate limited — sleeping 60s...")
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  [HTTP {resp.status_code}]")
            return None

        soup    = BeautifulSoup(resp.text, "html.parser")
        results = {}

        # Extract final scores from scorebox
        scorebox = soup.find("div", {"class": "scorebox"})
        scores = {}
        if scorebox:
            score_divs = scorebox.find_all("div", {"class": "score"})
            team_links = []
            for strong in scorebox.find_all("strong"):
                a = strong.find("a", href=True)
                if a and "/teams/" in a["href"]:
                    parts = a["href"].split("/")
                    if len(parts) >= 3:
                        team_links.append(parts[2].upper())
            if len(score_divs) >= 2 and len(team_links) >= 2:
                try:
                    scores[team_links[0]] = int(score_divs[0].get_text(strip=True))  # away
                    scores[team_links[1]] = int(score_divs[1].get_text(strip=True))  # home
                except:
                    pass

        for team_abbrev in [away_abbrev, home_abbrev]:
            skater_table = soup.find("table", {"id": f"{team_abbrev}_skaters"})
            adv_table    = soup.find("table", {"id": f"{team_abbrev}_adv_ALLAll"})

            if not skater_table:
                print(f"  [WARN] No skater table for {team_abbrev}")
                continue

            # Use data-stat names — robust to column order changes
            shots    = sum_col_by_stat(skater_table, "shots")
            pim      = sum_col_by_stat(skater_table, "pen_min")
            pp_goals = sum_col_by_stat(skater_table, "goals_pp")

            # Advanced table: HIT and BLK by data-stat
            hits          = sum_col_by_stat(adv_table, "hits")
            blocked_shots = sum_col_by_stat(adv_table, "blocks")

            results[team_abbrev] = {
                "shots":         shots,
                "hits":          hits,
                "pp_goals":      pp_goals,
                "pim":           pim,
                "blocked_shots": blocked_shots,
            }

        return {"stats": results, "scores": scores} if len(results) >= 2 else None

    except requests.exceptions.ConnectionError as e:
        print(f"  [CONNECTION ERROR] {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

def safe_int(val):
    try:
        return int(val) if val is not None else None
    except:
        return None

# ============================================================
# Main scraping loop
# ============================================================

stat_id  = con.execute("SELECT COALESCE(MAX(StatID), 0) FROM GameStats").fetchone()[0] + 1
inserted = 0
failed   = 0

for i, (game_id, game_date, home_id, away_id) in enumerate(games):
    home_name = team_id_to_name.get(home_id, "Unknown")
    away_name = team_id_to_name.get(away_id, "Unknown")
    home_abbr = team_id_to_abbrev.get(home_id, "???")
    away_abbr = team_id_to_abbrev.get(away_id, "???")
    url       = build_url(game_date, home_id)

    print(f"[{i+1}/{len(games)}] {str(game_date)[:10]} | {away_name} @ {home_name}")

    result = scrape_box_score(url, home_abbr, away_abbr)

    if not result:
        print(f"  [SKIP] Could not scrape stats")
        failed += 1
    else:
        stats  = result["stats"]
        scores = result["scores"]

        # Update Games table with actual scores
        if scores and home_abbr in scores and away_abbr in scores:
            home_score = scores[home_abbr]
            away_score = scores[away_abbr]
            winner_id  = home_id if home_score > away_score else away_id
            con.execute(f"""
                UPDATE Games SET
                    HomeScore    = {home_score},
                    AwayScore    = {away_score},
                    WinnerTeamID = {winner_id}
                WHERE GameID = {game_id}
            """)
            print(f"  Score: {away_abbr} {away_score} — {home_score} {home_abbr}")

        for team_id, abbrev in [(away_id, away_abbr), (home_id, home_abbr)]:
            team_stats = stats.get(abbrev)
            if not team_stats:
                print(f"  [WARN] No stats for {abbrev}")
                continue
            try:
                con.execute(
                    "INSERT INTO GameStats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        stat_id,
                        game_id,
                        team_id,
                        safe_int(team_stats.get("shots")),
                        safe_int(team_stats.get("hits")),
                        safe_int(team_stats.get("pp_goals")),
                        safe_int(team_stats.get("pim")),
                        None,  # FaceoffWinPct — not in box score
                        safe_int(team_stats.get("blocked_shots")),
                        None,  # Giveaways — not in box score
                        None,  # Takeaways — not in box score
                    )
                )
                stat_id  += 1
                inserted += 1
                print(f"  {abbrev}: Shots={team_stats.get('shots')} Hits={team_stats.get('hits')} PPG={team_stats.get('pp_goals')} PIM={team_stats.get('pim')} BLK={team_stats.get('blocked_shots')}")
            except Exception as e:
                print(f"  [ERROR] Inserting {abbrev}: {e}")

    if (i + 1) % 50 == 0:
        print(f"\n  --- Progress: {inserted} rows inserted ({inserted//2} games) ---\n")

    time.sleep(random.uniform(4.0, 6.0))

# ============================================================
# Summary
# ============================================================
print(f"\n=== Scraping Complete ===")
print(f"  Games attempted:  {len(games)}")
print(f"  Rows inserted:    {inserted} ({inserted//2} games)")
print(f"  Games failed:     {failed}")
total = con.execute("SELECT COUNT(*) FROM GameStats").fetchone()[0]
print(f"  Total in DB:      {total} GameStats rows")
con.close()
print("\nDone!")
