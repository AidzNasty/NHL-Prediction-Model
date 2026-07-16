"""
update_playoff_scores.py
Scrapes hockey-reference for 2025-26 playoff game results and updates
any Games rows that have NULL scores (games already seeded but not yet scored).
Run: python update_playoff_scores.py
"""
import os, time, requests, duckdb
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.hockey-reference.com/",
}

# HR uses different abbreviations for some teams
HR_ABBREV_MAP = {
    "L.A": "LAK", "T.B": "TBL", "N.J": "NJD", "S.J": "SJS",
    "VGK": "VGK", "UTA": "UTA",
}

def normalize_abbrev(hr_abbrev):
    return HR_ABBREV_MAP.get(hr_abbrev.upper(), hr_abbrev.upper())

def fetch_page(url, delay=4):
    time.sleep(delay)
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code == 429:
        print("  [429] rate-limited — sleeping 60s...")
        time.sleep(60)
        r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        print(f"  [HTTP {r.status_code}] {url}")
        return None
    return BeautifulSoup(r.text, "html.parser")

print(f"Connecting to MotherDuck ({DB})...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

# Load teams
teams_rows = con.execute("SELECT TeamID, TeamName, Abbreviation FROM Teams").fetchall()
by_abbrev  = {r[2].upper(): r[0] for r in teams_rows}
by_name    = {r[1]: r[0] for r in teams_rows}

# Find all 2025-26 playoff games with NULL scores
null_games = con.execute("""
    SELECT g.GameID, g.GameDate, t1.TeamName AS Away, t2.TeamName AS Home,
           t1.Abbreviation AS AwayAbbr, t2.Abbreviation AS HomeAbbr
    FROM Games g
    JOIN Teams t1 ON g.AwayTeamID = t1.TeamID
    JOIN Teams t2 ON g.HomeTeamID = t2.TeamID
    WHERE g.GameType = 'Playoffs'
      AND g.Season   = '2025-26'
      AND g.HomeScore IS NULL
    ORDER BY g.GameDate
""").fetchall()

if not null_games:
    print("No playoff games with NULL scores found. Nothing to update.")
    con.close()
    exit()

print(f"Found {len(null_games)} playoff game(s) with missing scores:")
for g in null_games:
    print(f"  {g[1]}  {g[2]} @ {g[3]}  (GameID={g[0]})")

# Fetch the 2025-26 playoffs index page
print("\nFetching hockey-reference playoff page...")
soup = fetch_page("https://www.hockey-reference.com/playoffs/NHL_2026.html")
if not soup:
    print("ERROR: Could not fetch playoff page.")
    con.close()
    exit()

# Collect all boxscore links
box_links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if "/boxscores/" in href and href.endswith(".html"):
        if href not in box_links:
            box_links.append(href)
print(f"Found {len(box_links)} boxscore links on playoff page.\n")

# Build a lookup: (date_str, home_abbr, away_abbr) -> box link
# so we can quickly find the right boxscore for each missing game
link_index = {}
for link in box_links:
    filename = link.split("/")[-1].replace(".html", "")
    try:
        game_date = datetime.strptime(filename[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except:
        continue
    # Extract team abbreviations from the filename (format: YYYYMMDDhhh0.html)
    # HR doesn't encode team abbrevs in the filename — we'll parse the page instead
    link_index[game_date] = link_index.get(game_date, []) + [link]

updated = 0
errors  = 0

for game_id, game_date_obj, away_name, home_name, away_abbr, home_abbr in null_games:
    game_date = str(game_date_obj)
    candidates = link_index.get(game_date, [])

    if not candidates:
        print(f"  [SKIP] No boxscore links found for {game_date}")
        errors += 1
        continue

    matched_url = None
    away_score = home_score = winner_id = None

    for link in candidates:
        url   = "https://www.hockey-reference.com" + link
        soup2 = fetch_page(url, delay=3)
        if not soup2:
            continue

        scorebox = soup2.find("div", {"class": "scorebox"})
        if not scorebox:
            continue

        # Extract team abbreviations from team hrefs
        team_hrefs = []
        for strong in scorebox.find_all("strong"):
            a = strong.find("a", href=True)
            if a and "/teams/" in a["href"]:
                parts = a["href"].split("/")
                if len(parts) >= 3:
                    team_hrefs.append((normalize_abbrev(parts[2]), a.get_text(strip=True)))

        if len(team_hrefs) < 2:
            continue

        pg_away_abbr, pg_away_name = team_hrefs[0]
        pg_home_abbr, pg_home_name = team_hrefs[1]

        # Match against our NULL game — compare abbreviations or names
        if not (
            (pg_away_abbr == away_abbr or pg_away_name == away_name) and
            (pg_home_abbr == home_abbr or pg_home_name == home_name)
        ):
            continue

        # Found the right boxscore — extract scores
        scores = scorebox.find_all("div", {"class": "score"})
        if len(scores) < 2:
            continue

        try:
            away_score = int(scores[0].get_text(strip=True))
            home_score = int(scores[1].get_text(strip=True))
        except:
            continue

        matched_url = url
        break

    if matched_url is None or away_score is None:
        print(f"  [SKIP] Could not match/parse boxscore for {game_date} {away_name} @ {home_name}")
        errors += 1
        continue

    # Determine winner
    home_id = by_name.get(home_name) or by_abbrev.get(home_abbr)
    away_id = by_name.get(away_name) or by_abbrev.get(away_abbr)
    winner_id = home_id if home_score > away_score else away_id

    # Update DB
    con.execute("""
        UPDATE Games
        SET HomeScore    = ?,
            AwayScore    = ?,
            WinnerTeamID = ?
        WHERE GameID = ?
    """, [home_score, away_score, winner_id, game_id])

    ot_str = ""  # OT detection could be added if needed
    print(f"  [UPDATED] {game_date} {away_name} @ {home_name}: {away_score}-{home_score} | Winner: {'Home' if winner_id == home_id else 'Away'}")
    updated += 1

print(f"\n{'='*55}")
print(f"  Updated: {updated}  |  Skipped/Errors: {errors}")
print(f"{'='*55}")
con.close()
print("\nDone! Now re-run app.py or refresh Streamlit.")
