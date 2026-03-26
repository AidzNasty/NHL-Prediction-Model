"""
update_player_accuracy.py
--------------------------
Tracks actual player performance vs predictions.

For each completed game where we have PlayerPredictions,
checks the NHL gamecenter API for actual goals/assists/points
and updates PlayerPredictions with actual results.

Also adds accuracy columns to PlayerPredictions table if missing.

Run daily after game results are in (add to 1:00 AM Task Scheduler job).

Usage: python update_player_accuracy.py
"""

import os
import time
import requests
import duckdb
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

# ── Add accuracy columns if missing ──────────────────────────
print("Checking PlayerPredictions schema...")
existing_cols = [r[0].lower() for r in con.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'PlayerPredictions'
""").fetchall()]

new_cols = [
    ("ActualGoals",   "INTEGER"),
    ("ActualAssists", "INTEGER"),
    ("ActualPoints",  "INTEGER"),
    ("GoalCorrect",   "INTEGER"),  # 1 if predicted > 20% and scored, 0 if predicted > 20% and didn't
    ("PointCorrect",  "INTEGER"),  # 1 if predicted > 40% and got point
    ("UpdatedDate",   "DATE"),
]

for col, dtype in new_cols:
    if col.lower() not in existing_cols:
        con.execute(f"ALTER TABLE PlayerPredictions ADD COLUMN {col} {dtype}")
        print(f"  Added: {col}")
    else:
        print(f"  Already exists: {col}")

# ── Get pending player predictions ───────────────────────────
print("\nFinding completed games with pending player predictions...")
pending = con.execute("""
    SELECT DISTINCT
        pp.GameID,
        g.GameDate,
        g.HomeTeamID,
        g.AwayTeamID
    FROM PlayerPredictions pp
    JOIN Games g ON pp.GameID = g.GameID
    WHERE pp.ActualGoals IS NULL
      AND g.HomeScore IS NOT NULL
      AND g.AwayScore IS NOT NULL
      AND g.GameDate >= '2026-01-01'
    ORDER BY g.GameDate DESC
    LIMIT 30
""").fetchall()

print(f"  Games to update: {len(pending)}")

total_updated = 0

for game_id, game_date, home_id, away_id in pending:
    # Find the NHL API game ID for this game
    # Games table doesn't store NHL API IDs so we need to find it from schedule
    date_str = str(game_date)

    try:
        schedule_resp = requests.get(
            f"https://api-web.nhle.com/v1/schedule/{date_str}",
            headers=HEADERS, timeout=10
        )
        if schedule_resp.status_code != 200:
            continue

        schedule = schedule_resp.json()

        # Find matching game
        nhl_game_id = None
        for week in schedule.get("gameWeek", []):
            if str(week.get("date",""))[:10] == date_str:
                for g in week.get("games", []):
                    home_abbrev = g.get("homeTeam", {}).get("abbrev", "")
                    away_abbrev = g.get("awayTeam", {}).get("abbrev", "")

                    # Match to our team IDs
                    home_match = con.execute(f"""
                        SELECT COUNT(*) FROM Teams
                        WHERE TeamID = {home_id}
                          AND (Abbreviation = '{home_abbrev}'
                               OR Abbreviation = '{
                                   home_abbrev.replace("NJD","N.J")
                                              .replace("SJS","S.J")
                                              .replace("TBL","T.B")
                                              .replace("LAK","L.A")
                               }')
                    """).fetchone()[0]

                    if home_match > 0:
                        nhl_game_id = g.get("id")
                        break

            if nhl_game_id:
                break

        if not nhl_game_id:
            continue

        # Get boxscore
        box_resp = requests.get(
            f"https://api-web.nhle.com/v1/gamecenter/{nhl_game_id}/boxscore",
            headers=HEADERS, timeout=10
        )
        if box_resp.status_code != 200:
            continue

        box = box_resp.json()
        pbg = box.get("playerByGameStats", {})
        if not pbg:
            continue

        # Build player stats lookup: NHL player ID → stats
        # We need to match by name since we don't store NHL player IDs
        player_stats = {}

        for side in ["homeTeam", "awayTeam"]:
            side_data = pbg.get(side, {})
            for group in ["forwards", "defense", "goalies"]:
                for p in side_data.get(group, []):
                    name = p.get("name", {}).get("default", "")
                    goals   = p.get("goals", 0) or 0
                    assists = p.get("assists", 0) or 0
                    points  = goals + assists
                    player_stats[name.lower()] = {
                        "goals":   goals,
                        "assists": assists,
                        "points":  points,
                    }

        # Get all player predictions for this game
        preds = con.execute(f"""
            SELECT pp.PlayerPredictionID, pp.PlayerID,
                   pp.GoalProbability, pp.AssistProbability,
                   pp.PointProbability,
                   p.FirstName || ' ' || p.LastName AS PlayerName
            FROM PlayerPredictions pp
            JOIN Players p ON pp.PlayerID = p.PlayerID
            WHERE pp.GameID = {game_id}
              AND pp.ActualGoals IS NULL
        """).fetchall()

        game_updated = 0
        for pred_id, player_id, goal_prob, assist_prob, point_prob, name in preds:
            # Match player name
            norm_name = name.lower()
            # Try full name first, then last name only
            stats = player_stats.get(norm_name)

            if not stats:
                # Try abbreviated name match (API uses "F. Lastname")
                last = norm_name.split()[-1] if norm_name else ""
                for api_name, api_stats in player_stats.items():
                    if api_name.endswith(last) and last:
                        stats = api_stats
                        break

            if not stats:
                continue

            actual_goals   = stats["goals"]
            actual_assists = stats["assists"]
            actual_points  = stats["points"]

            # Accuracy: did high-confidence predictions come true?
            goal_correct  = None
            point_correct = None

            if goal_prob is not None and goal_prob >= 0.20:
                goal_correct = 1 if actual_goals > 0 else 0

            if point_prob is not None and point_prob >= 0.40:
                point_correct = 1 if actual_points > 0 else 0

            con.execute(f"""
                UPDATE PlayerPredictions SET
                    ActualGoals   = ?,
                    ActualAssists = ?,
                    ActualPoints  = ?,
                    GoalCorrect   = ?,
                    PointCorrect  = ?,
                    UpdatedDate   = ?
                WHERE PlayerPredictionID = {pred_id}
            """, [actual_goals, actual_assists, actual_points,
                  goal_correct, point_correct, date.today()])

            game_updated  += 1
            total_updated += 1

        if game_updated > 0:
            home_name = con.execute(
                f"SELECT TeamName FROM Teams WHERE TeamID = {home_id}"
            ).fetchone()[0]
            away_name = con.execute(
                f"SELECT TeamName FROM Teams WHERE TeamID = {away_id}"
            ).fetchone()[0]
            print(f"  ✓ {game_date} {away_name} @ {home_name}: "
                  f"{game_updated} players updated")

        time.sleep(0.5)

    except Exception as e:
        print(f"  [ERROR] Game {game_id}: {e}")
        continue

# ── Accuracy Summary ──────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Players updated: {total_updated}")

# Overall accuracy stats
stats = con.execute("""
    SELECT
        COUNT(*)                                           AS total_preds,
        SUM(CASE WHEN ActualGoals IS NOT NULL THEN 1 END) AS with_actuals,
        -- Goal accuracy (for predictions >= 20%)
        SUM(CASE WHEN GoalCorrect = 1 THEN 1 ELSE 0 END)  AS goal_correct,
        SUM(CASE WHEN GoalCorrect IS NOT NULL THEN 1 END)  AS goal_total,
        -- Point accuracy (for predictions >= 40%)
        SUM(CASE WHEN PointCorrect = 1 THEN 1 ELSE 0 END) AS point_correct,
        SUM(CASE WHEN PointCorrect IS NOT NULL THEN 1 END) AS point_total
    FROM PlayerPredictions
""").fetchone()

total, with_actuals, gc, gt, pc, pt = stats
print(f"  Total predictions:    {total}")
print(f"  With actual results:  {with_actuals}")

if gt and gt > 0:
    print(f"  Goal accuracy:        {gc}/{gt} ({gc/gt*100:.1f}%)"
          f" [when predicted ≥20%]")
if pt and pt > 0:
    print(f"  Point accuracy:       {pc}/{pt} ({pc/pt*100:.1f}%)"
          f" [when predicted ≥40%]")

print(f"{'='*55}")
con.close()
print("\nDone!")
