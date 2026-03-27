"""
scrape_playoffs.py — Fixed duplicate check version
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

ABBREV_MAP = {
    "ANA":"ANA","BOS":"BOS","BUF":"BUF","CGY":"CGY","CAR":"CAR",
    "CHI":"CHI","COL":"COL","CBJ":"CBJ","DAL":"DAL","DET":"DET",
    "EDM":"EDM","FLA":"FLA","LAK":"L.A","MIN":"MIN","MTL":"MTL",
    "NSH":"NSH","NJD":"N.J","NYI":"NYI","NYR":"NYR","OTT":"OTT",
    "PHI":"PHI","PIT":"PIT","SJS":"S.J","SEA":"SEA","STL":"STL",
    "TBL":"T.B","TOR":"TOR","UTA":"UTA","VAN":"VAN","VEG":"VGK",
    "WSH":"WSH","WPG":"WPG",
}

def safe_int(val):
    try:
        return int(str(val).replace(",","")) if val and str(val).strip() not in ("","-") else None
    except:
        return None

def fetch(url, delay=4):
    time.sleep(delay)
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code == 429:
        print("  [429] sleeping 60s...")
        time.sleep(60)
        r = requests.get(url, headers=HEADERS, timeout=20)
    return BeautifulSoup(r.text, "html.parser") if r.status_code == 200 else None

print(f"Connecting...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

teams     = con.execute("SELECT TeamID, TeamName, Abbreviation FROM Teams").fetchall()
by_abbrev = {r[2]: r[0] for r in teams}
by_name   = {r[1]: r[0] for r in teams}

max_game_id = con.execute("SELECT COALESCE(MAX(GameID),0) FROM Games").fetchone()[0] + 1
max_stat_id = con.execute("SELECT COALESCE(MAX(StatID),0) FROM GameStats").fetchone()[0] + 1

print("Fetching HR playoff page...")
soup = fetch("https://www.hockey-reference.com/playoffs/NHL_2025.html")
box_links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if "/boxscores/" in href and href.endswith(".html") and "2025" in href:
        if href not in box_links:
            box_links.append(href)
print(f"Found {len(box_links)} links\n")

inserted = skipped = errors = 0

for link in box_links:
    url      = "https://www.hockey-reference.com" + link
    filename = link.split("/")[-1].replace(".html","")

    try:
        game_date = datetime.strptime(filename[:8], "%Y%m%d").date()
    except:
        skipped += 1
        continue

    # Fetch page first to get both teams
    soup2 = fetch(url)
    if not soup2:
        errors += 1
        continue

    scorebox = soup2.find("div", {"class": "scorebox"})
    if not scorebox:
        errors += 1
        continue

    # Extract teams from strong links
    team_hrefs = []
    for strong in scorebox.find_all("strong"):
        a = strong.find("a", href=True)
        if a and "/teams/" in a["href"]:
            parts = a["href"].split("/")
            if len(parts) >= 3:
                team_hrefs.append((parts[2].upper(), a.get_text(strip=True)))

    if len(team_hrefs) < 2:
        errors += 1
        continue

    away_hr, away_name = team_hrefs[0]
    home_hr, home_name = team_hrefs[1]

    away_team_id = by_abbrev.get(ABBREV_MAP.get(away_hr, away_hr)) or by_name.get(away_name)
    home_team_id = by_abbrev.get(ABBREV_MAP.get(home_hr, home_hr)) or by_name.get(home_name)

    if not away_team_id or not home_team_id:
        print(f"  [NO TEAM] {away_hr}/{home_hr} — {away_name}/{home_name}")
        errors += 1
        continue

    # Check duplicate using both teams
    existing = con.execute(f"""
        SELECT GameID FROM Games
        WHERE GameDate   = '{game_date}'
          AND HomeTeamID = {home_team_id}
          AND AwayTeamID = {away_team_id}
          AND GameType   = 'Playoffs'
    """).fetchone()

    if existing:
        skipped += 1
        continue

    # Scores
    scores = scorebox.find_all("div", {"class": "score"})
    if len(scores) < 2:
        errors += 1
        continue

    try:
        away_score = int(scores[0].get_text(strip=True))
        home_score = int(scores[1].get_text(strip=True))
    except:
        errors += 1
        continue

    # OT check
    is_ot = False
    meta = scorebox.find("div", {"class": "scorebox_meta"})
    if meta and any(x in meta.get_text().lower()
                    for x in ["overtime","shootout"," ot"," so"]):
        is_ot = True
    if not is_ot:
        linescore = soup2.find("table", {"id": "linescore"})
        if linescore:
            hdrs = [th.get_text(strip=True)
                    for th in linescore.find("thead").find_all("th")]
            for row in linescore.find("tbody").find_all("tr"):
                cells = [td.get_text(strip=True)
                         for td in row.find_all("td")]
                for i, h in enumerate(hdrs):
                    if h in ("OT","2OT","3OT","SO") and i < len(cells):
                        if cells[i] and cells[i] != "0":
                            is_ot = True

    winner_id = home_team_id if home_score > away_score else away_team_id

    home_b2b = con.execute(f"""
        SELECT COUNT(*) FROM Games
        WHERE (HomeTeamID={home_team_id} OR AwayTeamID={home_team_id})
          AND GameDate='{game_date}'::DATE - INTERVAL '1 day'
    """).fetchone()[0] > 0

    away_b2b = con.execute(f"""
        SELECT COUNT(*) FROM Games
        WHERE (HomeTeamID={away_team_id} OR AwayTeamID={away_team_id})
          AND GameDate='{game_date}'::DATE - INTERVAL '1 day'
    """).fetchone()[0] > 0

    con.execute("""
        INSERT INTO Games (
            GameID,HomeTeamID,AwayTeamID,GameDate,Season,
            GameType,HomeScore,AwayScore,WinnerTeamID,
            OvertimeFlag,HomeIsBackToBack,AwayIsBackToBack
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, [max_game_id, home_team_id, away_team_id, game_date,
          "2024-25", "Playoffs", home_score, away_score,
          winner_id, is_ot, home_b2b, away_b2b])

    game_id = max_game_id
    max_game_id += 1

    # GameStats for both teams
    for team_id, hr_abbrev in [(home_team_id, home_hr),
                               (away_team_id, away_hr)]:
        shots = hits = pim = blk = None
        for tid in [f"stats_{hr_abbrev}", f"stats_{hr_abbrev.lower()}"]:
            tbl = soup2.find("table", {"id": tid})
            if tbl:
                tfoot = tbl.find("tfoot")
                if tfoot:
                    row = tfoot.find("tr")
                    if row:
                        cells = {td.get("data-stat"): td.get_text(strip=True)
                                 for td in row.find_all("td")}
                        shots = safe_int(cells.get("shots"))
                        hits  = safe_int(cells.get("hits"))
                        pim   = safe_int(cells.get("pen_min"))
                        blk   = safe_int(cells.get("blocked_shots") or
                                         cells.get("blk"))
                break

        con.execute("""
            INSERT INTO GameStats (
                StatID,GameID,TeamID,Shots,Hits,PowerPlayGoals,
                PenaltyMinutes,BlockedShots,FaceoffWinPct,Giveaways,Takeaways
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, [max_stat_id, game_id, team_id,
              shots, hits, None, pim, blk, None, None, None])
        max_stat_id += 1

    ot_str = " (OT/SO)" if is_ot else ""
    print(f"  OK {game_date} {away_name} @ {home_name}: {away_score}-{home_score}{ot_str}")
    inserted += 1

total = con.execute("SELECT COUNT(*) FROM Games WHERE GameType='Playoffs'").fetchone()[0]
print(f"\n{'='*55}")
print(f"  Inserted: {inserted} | Skipped: {skipped} | Errors: {errors}")
print(f"  Total playoff games in DB: {total}")
print(f"{'='*55}")
con.close()
print("\nDone!")
