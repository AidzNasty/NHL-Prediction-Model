"""
update_calculated_fields.py
----------------------------
Adds and populates calculated fields across three tables:

1. TeamStandings — home/away splits + faceoff/GV/TK tendencies
   Adds columns:
   - HomeGP, HomeW, HomeL, HomeOTL, HomeWinPct
   - AwayGP, AwayW, AwayL, AwayOTL, AwayWinPct
   - HomeGF_Per_Game, HomeGA_Per_Game
   - AwayGF_Per_Game, AwayGA_Per_Game
   - Team_FaceoffWinPct, Team_Giveaways_Per_Game, Team_Takeaways_Per_Game

2. Games — back-to-back flags
   Adds columns:
   - HomeIsBackToBack (BOOLEAN)
   - AwayIsBackToBack (BOOLEAN)

Usage: python update_calculated_fields.py
"""

import os
import duckdb
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

# ── 1. Add columns to TeamStandings ──────────────────────────
print("Step 1: Adding columns to TeamStandings...")

new_cols = [
    ("HomeGP",                  "INTEGER"),
    ("HomeW",                   "INTEGER"),
    ("HomeL",                   "INTEGER"),
    ("HomeOTL",                 "INTEGER"),
    ("HomeWinPct",              "DECIMAL(6,3)"),
    ("AwayGP",                  "INTEGER"),
    ("AwayW",                   "INTEGER"),
    ("AwayL",                   "INTEGER"),
    ("AwayOTL",                 "INTEGER"),
    ("AwayWinPct",              "DECIMAL(6,3)"),
    ("HomeGF_Per_Game",         "DECIMAL(5,2)"),
    ("HomeGA_Per_Game",         "DECIMAL(5,2)"),
    ("AwayGF_Per_Game",         "DECIMAL(5,2)"),
    ("AwayGA_Per_Game",         "DECIMAL(5,2)"),
    ("Team_FaceoffWinPct",      "DECIMAL(6,2)"),
    ("Team_Giveaways_Per_Game", "DECIMAL(6,2)"),
    ("Team_Takeaways_Per_Game", "DECIMAL(6,2)"),
]

existing_cols = [r[0].lower() for r in con.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'TeamStandings'
""").fetchall()]

for col, dtype in new_cols:
    if col.lower() not in existing_cols:
        con.execute(f"ALTER TABLE TeamStandings ADD COLUMN {col} {dtype}")
        print(f"  Added: {col}")
    else:
        print(f"  Already exists: {col}")

# ── 2. Add columns to Games ───────────────────────────────────
print("\nStep 2: Adding back-to-back columns to Games...")

games_cols = [
    ("HomeIsBackToBack", "BOOLEAN"),
    ("AwayIsBackToBack", "BOOLEAN"),
]

existing_game_cols = [r[0].lower() for r in con.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'Games'
""").fetchall()]

for col, dtype in games_cols:
    if col.lower() not in existing_game_cols:
        con.execute(f"ALTER TABLE Games ADD COLUMN {col} {dtype} DEFAULT FALSE")
        print(f"  Added: {col}")
    else:
        print(f"  Already exists: {col}")

# ── 3. Populate Home/Away splits in TeamStandings ────────────
print("\nStep 3: Populating home/away splits...")

for season in ["2025-26", "2024-25"]:
    print(f"  Season: {season}")

    # Get all teams
    teams = con.execute("SELECT TeamID FROM Teams").fetchall()

    for (team_id,) in teams:
        # Home stats
        home = con.execute(f"""
            SELECT
                COUNT(*)                                          AS HomeGP,
                SUM(CASE WHEN HomeScore > AwayScore THEN 1 ELSE 0 END) AS HomeW,
                SUM(CASE WHEN HomeScore < AwayScore AND OvertimeFlag = FALSE THEN 1 ELSE 0 END) AS HomeL,
                SUM(CASE WHEN HomeScore < AwayScore AND OvertimeFlag = TRUE  THEN 1 ELSE 0 END) AS HomeOTL,
                AVG(CAST(HomeScore AS DOUBLE))                    AS HomeGF_Per_Game,
                AVG(CAST(AwayScore AS DOUBLE))                    AS HomeGA_Per_Game
            FROM Games
            WHERE HomeTeamID = {team_id}
              AND Season = '{season}'
              AND HomeScore IS NOT NULL
        """).fetchone()

        # Away stats
        away = con.execute(f"""
            SELECT
                COUNT(*)                                          AS AwayGP,
                SUM(CASE WHEN AwayScore > HomeScore THEN 1 ELSE 0 END) AS AwayW,
                SUM(CASE WHEN AwayScore < HomeScore AND OvertimeFlag = FALSE THEN 1 ELSE 0 END) AS AwayL,
                SUM(CASE WHEN AwayScore < HomeScore AND OvertimeFlag = TRUE  THEN 1 ELSE 0 END) AS AwayOTL,
                AVG(CAST(AwayScore AS DOUBLE))                    AS AwayGF_Per_Game,
                AVG(CAST(HomeScore AS DOUBLE))                    AS AwayGA_Per_Game
            FROM Games
            WHERE AwayTeamID = {team_id}
              AND Season = '{season}'
              AND AwayScore IS NOT NULL
        """).fetchone()

        home_gp, home_w, home_l, home_otl, home_gf, home_ga = home
        away_gp, away_w, away_l, away_otl, away_gf, away_ga = away

        home_win_pct = round(home_w / home_gp, 3) if home_gp and home_gp > 0 else None
        away_win_pct = round(away_w / away_gp, 3) if away_gp and away_gp > 0 else None

        # Faceoff, GV, TK from PlayerStats (season tendency)
        fo_gv_tk = con.execute(f"""
            SELECT
                SUM(Faceoffs_Won) * 100.0 / NULLIF(SUM(Faceoffs_Won) + SUM(Faceoffs_Lost), 0) AS FO_Pct,
                SUM(Giveaways) * 1.0 / NULLIF(MAX(gp.GP), 0) AS GV_Per_Game,
                SUM(Takeaways) * 1.0 / NULLIF(MAX(gp.GP), 0) AS TK_Per_Game
            FROM PlayerStats ps
            JOIN Players p ON ps.PlayerID = p.PlayerID
            CROSS JOIN (
                SELECT COALESCE(MAX(GP), 1) AS GP
                FROM TeamStandings
                WHERE TeamID = {team_id} AND Season = '{season}'
            ) gp
            WHERE p.TeamID = {team_id}
              AND ps.Season = '{season}'
              AND (ps.Faceoffs_Won IS NOT NULL OR ps.Giveaways IS NOT NULL OR ps.Takeaways IS NOT NULL)
        """).fetchone()

        fo_pct, gv_per_game, tk_per_game = fo_gv_tk if fo_gv_tk else (None, None, None)

        con.execute(f"""
            UPDATE TeamStandings SET
                HomeGP                  = ?,
                HomeW                   = ?,
                HomeL                   = ?,
                HomeOTL                 = ?,
                HomeWinPct              = ?,
                AwayGP                  = ?,
                AwayW                   = ?,
                AwayL                   = ?,
                AwayOTL                 = ?,
                AwayWinPct              = ?,
                HomeGF_Per_Game         = ?,
                HomeGA_Per_Game         = ?,
                AwayGF_Per_Game         = ?,
                AwayGA_Per_Game         = ?,
                Team_FaceoffWinPct      = ?,
                Team_Giveaways_Per_Game = ?,
                Team_Takeaways_Per_Game = ?
            WHERE TeamID = {team_id} AND Season = '{season}'
        """, [
            home_gp, home_w, home_l, home_otl, home_win_pct,
            away_gp, away_w, away_l, away_otl, away_win_pct,
            round(home_gf, 2) if home_gf else None,
            round(home_ga, 2) if home_ga else None,
            round(away_gf, 2) if away_gf else None,
            round(away_ga, 2) if away_ga else None,
            round(fo_pct,  2) if fo_pct  else None,
            round(gv_per_game, 2) if gv_per_game else None,
            round(tk_per_game, 2) if tk_per_game else None,
        ])

    updated = con.execute(f"""
        SELECT COUNT(*) FROM TeamStandings
        WHERE Season = '{season}' AND HomeWinPct IS NOT NULL
    """).fetchone()[0]
    print(f"    Updated {updated} teams")

# ── 4. Populate back-to-back flags in Games ───────────────────
print("\nStep 4: Populating back-to-back flags...")

for season in ["2025-26", "2024-25"]:
    print(f"  Season: {season}")

    # Get all completed games for the season
    games = con.execute(f"""
        SELECT GameID, GameDate, HomeTeamID, AwayTeamID
        FROM Games
        WHERE Season = '{season}'
        ORDER BY GameDate ASC
    """).fetchall()

    # Build set of (team, date) for fast lookup
    played = {}  # team_id -> sorted list of dates played
    for game_id, game_date, home_id, away_id in games:
        date_str = str(game_date)[:10]
        played.setdefault(home_id, []).append(date_str)
        played.setdefault(away_id, []).append(date_str)

    b2b_count = 0
    for game_id, game_date, home_id, away_id in games:
        date_str = str(game_date)[:10]
        from datetime import datetime, timedelta
        game_dt   = datetime.strptime(date_str, "%Y-%m-%d")
        yesterday = (game_dt - timedelta(days=1)).strftime("%Y-%m-%d")

        home_b2b = yesterday in played.get(home_id, [])
        away_b2b = yesterday in played.get(away_id, [])

        if home_b2b or away_b2b:
            con.execute("""
                UPDATE Games
                SET HomeIsBackToBack = ?, AwayIsBackToBack = ?
                WHERE GameID = ?
            """, [home_b2b, away_b2b, game_id])
            b2b_count += 1

    print(f"    {b2b_count} back-to-back games flagged")

# ── Summary ───────────────────────────────────────────────────
print("\n=== Complete ===")
sample = con.execute("""
    SELECT TeamID, Season, HomeWinPct, AwayWinPct,
           HomeGF_Per_Game, AwayGF_Per_Game,
           Team_FaceoffWinPct, Team_Giveaways_Per_Game
    FROM TeamStandings
    WHERE HomeWinPct IS NOT NULL
    LIMIT 3
""").fetchall()
print("\nSample TeamStandings (calculated fields):")
for r in sample:
    print(f"  Team {r[0]} {r[1]}: HomeW%={r[2]} AwayW%={r[3]} HomeGF={r[4]} FO%={r[6]} GV/G={r[7]}")

b2b_total = con.execute("SELECT COUNT(*) FROM Games WHERE HomeIsBackToBack = TRUE OR AwayIsBackToBack = TRUE").fetchone()[0]
print(f"\nTotal back-to-back game instances: {b2b_total}")

con.close()
print("\nDone!")
