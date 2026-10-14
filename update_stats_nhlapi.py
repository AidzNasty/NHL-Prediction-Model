"""
update_stats_nhlapi.py
-----------------------
Refreshes TeamStandings, PlayerStats and GoalieStats for a season from the
official NHL API. Used because Natural Stat Trick (scrape_nst.py) now returns
HTTP 403 to scripted requests.

The NHL API has no expected-goals / scoring-chance / danger-zone data, so those
NST-only columns are left NULL (the models fill NULL features with 0).
CF_Pct comes from the NHL's shot-attempt percentage (SAT%), which is Corsi.

Run update_calculated_fields.py afterwards to fill home/away splits.

Usage:
    python update_stats_nhlapi.py                    # 2026-27
    python update_stats_nhlapi.py --season 2026-27
"""

import os
import sys
import argparse
import unicodedata
import requests
import duckdb
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
load_dotenv()

TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

HEADERS = {"User-Agent": "Mozilla/5.0"}
STATS   = "https://api.nhle.com/stats/rest/en/"

parser = argparse.ArgumentParser()
parser.add_argument("--season", default="2026-27", help="Season e.g. 2026-27")
args = parser.parse_args()

season     = args.season
api_season = f"{season[:4]}20{season[5:]}"   # 2026-27 -> 20262027


def normalize(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


def stats_api(endpoint):
    url  = f"{STATS}{endpoint}?limit=-1&cayenneExp=seasonId={api_season}%20and%20gameTypeId=2"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def pct(a, b):
    return round(100.0 * a / (a + b), 2) if a is not None and b is not None and (a + b) > 0 else None


print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

teams        = con.execute("SELECT TeamID, TeamName, Abbreviation FROM Teams").fetchall()
team_by_abbr = {r[2].upper(): r[0] for r in teams}
team_by_name = {normalize(r[1]): r[0] for r in teams}

# ── 1. TeamStandings ──────────────────────────────────────────
print(f"Team standings — {season}")
standings = requests.get("https://api-web.nhle.com/v1/standings/now", headers=HEADERS, timeout=30).json()["standings"]
summary   = {team_by_name.get(normalize(r["teamFullName"])): r for r in stats_api("team/summary")}
realtime  = {team_by_name.get(normalize(r["teamFullName"])): r for r in stats_api("team/realtime")}

con.execute("DELETE FROM TeamStandings WHERE Season = ?", [season])
standing_id = con.execute("SELECT COALESCE(MAX(StandingID), 0) FROM TeamStandings").fetchone()[0]

for s in standings:
    team_id = team_by_abbr.get(s["teamAbbrev"]["default"].upper())
    if not team_id or str(s.get("seasonId")) != api_season:
        print(f"  [SKIP] {s['teamAbbrev']['default']}")
        continue
    gp  = s["gamesPlayed"]
    gf, ga = s["goalFor"], s["goalAgainst"]
    sm  = summary.get(team_id, {})
    rt  = realtime.get(team_id, {})
    sf  = round(sm["shotsForPerGame"] * gp) if sm.get("shotsForPerGame") is not None else None
    sa  = round(sm["shotsAgainstPerGame"] * gp) if sm.get("shotsAgainstPerGame") is not None else None
    sh_pct = round(100.0 * gf / sf, 2) if sf else None
    sv_pct = round(100.0 * (sa - ga) / sa, 2) if sa else None
    pdo    = round((sh_pct + sv_pct) / 100.0, 3) if sh_pct is not None and sv_pct is not None else None

    standing_id += 1
    con.execute("""
        INSERT INTO TeamStandings (
            StandingID, TeamID, Season, GP, W, L, OTL, ROW, Points, Point_Pct,
            CF_Pct, SF, SA, SF_Pct, GF, GA, GF_Pct, SH_Pct, SV_Pct, PDO,
            Team_FaceoffWinPct, Team_Giveaways_Per_Game, Team_Takeaways_Per_Game
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        standing_id, team_id, season, gp, s["wins"], s["losses"], s["otLosses"],
        s["regulationPlusOtWins"], s["points"], round(s["pointPctg"], 3),
        round(rt["satPct"] * 100, 2) if rt.get("satPct") is not None else None,
        sf, sa, pct(sf, sa), gf, ga, pct(gf, ga), sh_pct, sv_pct, pdo,
        round(sm["faceoffWinPct"] * 100, 2) if sm.get("faceoffWinPct") is not None else None,
        round(rt["giveaways"] / gp, 2) if gp and rt.get("giveaways") is not None else None,
        round(rt["takeaways"] / gp, 2) if gp and rt.get("takeaways") is not None else None,
    ])
print(f"  {con.execute('SELECT COUNT(*) FROM TeamStandings WHERE Season = ?', [season]).fetchone()[0]} teams written\n")

# ── Player lookup ─────────────────────────────────────────────
rows = con.execute("""
    SELECT p.PlayerID, p.FirstName || ' ' || p.LastName, t.Abbreviation
    FROM Players p JOIN Teams t ON p.TeamID = t.TeamID
    WHERE p.IsActive = TRUE
""").fetchall()
by_name_team = {(normalize(r[1]), r[2].upper()): r[0] for r in rows}
by_name = {}
for r in rows:
    by_name.setdefault(normalize(r[1]), r[0])


def resolve(full_name, team_abbrevs):
    team = (team_abbrevs or "").split(",")[-1].strip().upper()   # most recent team
    key  = normalize(full_name)
    return by_name_team.get((key, team)) or by_name.get(key)


# ── 2. PlayerStats ────────────────────────────────────────────
print(f"Skater stats — {season}")
sk_rt = {r["playerId"]: r for r in stats_api("skater/realtime")}
sk_fo = {r["playerId"]: r for r in stats_api("skater/faceoffwins")}
cat_id = con.execute("SELECT CategoryID FROM StatCategories WHERE CategoryName='Skater Stats'").fetchone()
cat_id = cat_id[0] if cat_id else 2

con.execute("DELETE FROM PlayerStats WHERE Season = ?", [season])
stat_id  = con.execute("SELECT COALESCE(MAX(PlayerStatID), 0) FROM PlayerStats").fetchone()[0]
inserted = no_match = 0
seen     = set()

for r in stats_api("skater/summary"):
    pid = resolve(r["skaterFullName"], r["teamAbbrevs"])
    if not pid or pid in seen:
        no_match += not pid
        continue
    seen.add(pid)
    rt = sk_rt.get(r["playerId"], {})
    fo = sk_fo.get(r["playerId"], {})
    gp = r["gamesPlayed"]
    stat_id += 1
    con.execute("""
        INSERT INTO PlayerStats (
            PlayerStatID, PlayerID, CategoryID, Season, GP, TOI,
            Goals, Total_Assists, Total_Points, Shots, SH_Pct, iCF, PIM,
            Giveaways, Takeaways, Hits, Shots_Blocked,
            Faceoffs_Won, Faceoffs_Lost, Faceoffs_Pct
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        stat_id, pid, cat_id, season, gp,
        round(r["timeOnIcePerGame"] * gp / 60.0, 4) if r.get("timeOnIcePerGame") else None,
        r["goals"], r["assists"], r["points"], r["shots"],
        round(r["shootingPct"] * 100, 2) if r.get("shootingPct") is not None else None,
        rt.get("totalShotAttempts"), r["penaltyMinutes"],
        rt.get("giveaways"), rt.get("takeaways"), rt.get("hits"), rt.get("blockedShots"),
        fo.get("totalFaceoffWins"), fo.get("totalFaceoffLosses"),
        round(fo["faceoffWinPct"] * 100, 2) if fo.get("faceoffWinPct") is not None else 0,
    ])
    inserted += 1
print(f"  Skaters inserted: {inserted} | No player match: {no_match}\n")

# ── 3. GoalieStats ────────────────────────────────────────────
print(f"Goalie stats — {season}")
con.execute("DELETE FROM GoalieStats WHERE Season = ?", [season])
stat_id  = con.execute("SELECT COALESCE(MAX(GoalieStatID), 0) FROM GoalieStats").fetchone()[0]
inserted = no_match = 0
seen     = set()

for r in stats_api("goalie/summary"):
    pid = resolve(r["goalieFullName"], r["teamAbbrevs"])
    if not pid or pid in seen:
        no_match += not pid
        continue
    seen.add(pid)
    stat_id += 1
    con.execute("""
        INSERT INTO GoalieStats (
            GoalieStatID, PlayerID, Season, GP, TOI, SA, SV, GA, SV_Pct, GAA
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """, [
        stat_id, pid, season, r["gamesPlayed"], round(r["timeOnIce"] / 60.0, 4),
        r["shotsAgainst"], r["saves"], r["goalsAgainst"],
        round(r["savePct"], 3) if r.get("savePct") is not None else None,
        round(r["goalsAgainstAverage"], 2) if r.get("goalsAgainstAverage") is not None else None,
    ])
    inserted += 1
print(f"  Goalies inserted: {inserted} | No player match: {no_match}")

con.close()
print("\nDone! Now run: python update_calculated_fields.py")
