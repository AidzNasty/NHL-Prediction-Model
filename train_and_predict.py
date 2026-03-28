"""
train_and_predict.py
---------------------
Daily runner:
  1. Retrains player + team models
  2. Updates actual results for pending predictions
  3. Predicts today's games
  4. Writes to MotherDuck Predictions + PlayerPredictions tables

Usage:
    python train_and_predict.py                        # retrain + predict
    python train_and_predict.py --predict-only         # skip retrain
    python train_and_predict.py --date 2026-03-25      # specific date
"""

import os
import sys
import argparse
import requests
from datetime import datetime, date

from db_features import (
    get_connection, get_todays_games,
    get_team_features, get_player_features, get_goalie_features
)
from player_model import PlayerModel
from team_model   import TeamModel

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--predict-only", action="store_true")
parser.add_argument("--date", type=str, default=None)
args = parser.parse_args()

SEASON = "2025-26"

print("\n" + "="*70)
print("NHL DAILY TRAIN + PREDICT")
print("="*70)
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Mode: {'PREDICT ONLY' if args.predict_only else 'RETRAIN + PREDICT'}")

# ── Connect ───────────────────────────────────────────────────
print("\nConnecting to MotherDuck...")
con = get_connection()
print("Connected!")

# ── Step 1: Train / Load models ───────────────────────────────
player_model = PlayerModel()
team_model   = TeamModel()

if not args.predict_only:
    print("\n" + "-"*50)
    print("STEP 1: TRAINING MODELS")
    print("-"*50)
    player_model.train(con)
    team_model.train(con, player_model=player_model)
else:
    print("\nLoading saved models...")
    try:
        player_model.load()
        team_model.load()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Run without --predict-only first to train models")
        sys.exit(1)

# ── Step 2: Update actual results ────────────────────────────
print("\n" + "-"*50)
print("STEP 2: UPDATING ACTUAL RESULTS")
print("-"*50)

pending = con.execute("""
    SELECT PredictionID, GameID, HomeTeamID, AwayTeamID,
           PredictedWinner, PredictedHomeScore, PredictedAwayScore
    FROM Predictions
    WHERE ActualWinner IS NULL
      AND PredictedWinner IS NOT NULL
      AND GameID IS NOT NULL
""").fetchall()

print(f"  Pending predictions: {len(pending)}")
updated = 0

for pred_id, game_id, home_id, away_id, pred_winner, pred_home, pred_away in pending:
    result = con.execute(f"""
        SELECT HomeScore, AwayScore, OvertimeFlag
        FROM Games
        WHERE GameID = {game_id}
          AND HomeScore IS NOT NULL
          AND AwayScore IS NOT NULL
    """).fetchone()

    if not result:
        continue

    actual_home = int(result[0] or 0)
    actual_away = int(result[1] or 0)
    actual_ot   = bool(result[2])

    home_name = con.execute(
        f"SELECT TeamName FROM Teams WHERE TeamID = {home_id}"
    ).fetchone()[0]
    away_name = con.execute(
        f"SELECT TeamName FROM Teams WHERE TeamID = {away_id}"
    ).fetchone()[0]

    actual_winner = home_name if actual_home > actual_away else away_name
    ml_correct    = 1 if pred_winner == actual_winner else 0

    con.execute(f"""
        UPDATE Predictions SET
            ActualWinner          = ?,
            ActualHomeScore       = ?,
            ActualAwayScore       = ?,
            ActualOT              = ?,
            MLCorrect             = ?,
            ActualOutcomeCorrect  = ?
        WHERE PredictionID = {pred_id}
    """, [actual_winner, actual_home, actual_away,
          actual_ot, ml_correct, bool(ml_correct)])

    result_str  = f"{actual_away}-{actual_home}" + (" (OT)" if actual_ot else "")
    correct_str = "OK" if ml_correct else "X"
    print(f"  {correct_str} {away_name} @ {home_name}: {result_str} "
          f"(predicted: {pred_winner})")
    updated += 1

print(f"  Updated: {updated} games")

# ── NHL game state lookup ─────────────────────────────────────
_nhl_game_states = None  # cache so we only fetch schedule once per run

def get_game_states(target_date):
    """Returns dict of {(home_abbrev, away_abbrev): state} for today's games.
    States: PRE (not started), LIVE/CRIT (in progress), OFF/FINAL (finished).
    """
    global _nhl_game_states
    if _nhl_game_states is not None:
        return _nhl_game_states
    _nhl_game_states = {}
    try:
        resp = requests.get(
            f"https://api-web.nhle.com/v1/schedule/{target_date}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        if resp.status_code != 200:
            return _nhl_game_states
        for week in resp.json().get("gameWeek", []):
            if str(week.get("date", ""))[:10] == str(target_date):
                for g in week.get("games", []):
                    home = g.get("homeTeam", {}).get("abbrev", "")
                    away = g.get("awayTeam", {}).get("abbrev", "")
                    state = g.get("gameState", "PRE")
                    _nhl_game_states[(home, away)] = state
    except Exception:
        pass
    return _nhl_game_states

# ── Step 3: Predict today's games ────────────────────────────
print("\n" + "-"*50)
print("STEP 3: PREDICTING TODAY'S GAMES")
print("-"*50)

target_date = date.fromisoformat(args.date) if args.date else date.today()
print(f"  Date: {target_date}")

# Update player actuals for yesterday's games
print("\n  Updating player actuals...")
try:
    import subprocess, sys
    yesterday = target_date - __import__('datetime').timedelta(days=1)
    result = subprocess.run(
        [sys.executable, "update_player_accuracy.py"],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.stdout:
        print(result.stdout[-500:])  # last 500 chars
    if result.returncode != 0 and result.stderr:
        print(f"  Warning: {result.stderr[:200]}")
except Exception as e:
    print(f"  Could not update player actuals: {e}")

today_games = get_todays_games(con, target_date)
print(f"  Games found: {len(today_games)}")

if len(today_games) == 0:
    print("  No games scheduled — nothing to predict")
else:
    max_id  = con.execute(
        "SELECT COALESCE(MAX(PredictionID), 0) FROM Predictions"
    ).fetchone()[0]
    pred_id = max_id + 1

    # Pre-load all data once
    print("  Pre-loading data...")
    players_df = get_player_features(con, SEASON)
    teams_df   = get_team_features(con, SEASON)
    goalies_df = get_goalie_features(con, SEASON)
    print(f"  Loaded: {len(teams_df)} teams, {len(players_df)} players")

    predictions_made = 0

    for _, game in today_games.iterrows():
        home_id   = int(game["HomeTeamID"])
        away_id   = int(game["AwayTeamID"])
        game_id   = int(game["GameID"])
        home_name = game["HomeTeamName"]
        away_name = game["AwayTeamName"]
        home_b2b  = bool(game["HomeIsBackToBack"])
        away_b2b  = bool(game["AwayIsBackToBack"])

        print(f"\n  {away_name} @ {home_name}")

        # Check NHL API game state — only re-predict PRE (not yet started) games.
        # LIVE/CRIT = in progress, OFF/FINAL = finished → preserve those predictions.
        home_abbrev = con.execute(f"SELECT Abbreviation FROM Teams WHERE TeamID = {home_id}").fetchone()[0]
        away_abbrev = con.execute(f"SELECT Abbreviation FROM Teams WHERE TeamID = {away_id}").fetchone()[0]
        game_states = get_game_states(target_date)
        game_state  = game_states.get((home_abbrev, away_abbrev), "PRE")

        existing = con.execute(f"""
            SELECT PredictionID, ActualWinner FROM Predictions
            WHERE GameID = {game_id}
              AND PredictedWinner IS NOT NULL
        """).fetchone()
        if existing:
            if game_state in ("LIVE", "CRIT", "OFF", "FINAL"):
                print(f"    Game is {game_state} — keeping existing prediction")
                continue
            if existing[1] is not None:
                print(f"    Game already completed — skipping")
                continue
            # Game is PRE (not started) — delete and re-predict with fresh lineup data
            pred_id = existing[0]
            con.execute(f"DELETE FROM PlayerPredictions WHERE GameID = {game_id} AND PredictionDate = CAST('{target_date}' AS DATE)")
            con.execute(f"DELETE FROM Predictions WHERE PredictionID = {pred_id}")
            print(f"    Game is PRE — refreshing prediction with latest lineup data...")

        # Player predictions
        print(f"    Player predictions...")
        home_players, home_proj = player_model.predict_team_players(
            con, home_id, away_id, is_home=True,
            season=SEASON, b2b=home_b2b, top_n=10,
            players_df=players_df, teams_df=teams_df, goalies_df=goalies_df
        )
        away_players, away_proj = player_model.predict_team_players(
            con, away_id, home_id, is_home=False,
            season=SEASON, b2b=away_b2b, top_n=10,
            players_df=players_df, teams_df=teams_df, goalies_df=goalies_df
        )

        # Team prediction
        print(f"    Team prediction...")
        team_pred = team_model.predict_game(
            con, home_id, away_id, SEASON,
            home_b2b=home_b2b, away_b2b=away_b2b,
            home_proj_goals=home_proj,
            away_proj_goals=away_proj
        )

        if team_pred is None:
            print(f"    WARNING: Could not build features")
            continue

        winner = team_pred["predicted_winner"]
        conf   = team_pred["confidence"]
        hscore = team_pred["home_score"]
        ascore = team_pred["away_score"]
        ot_prob= team_pred["ot_prob"]

        b2b_flags = []
        if home_b2b: b2b_flags.append(f"{home_name} B2B")
        if away_b2b: b2b_flags.append(f"{away_name} B2B")
        b2b_str = f" | ⚠ {', '.join(b2b_flags)}" if b2b_flags else ""

        print(f"\n    ┌──────────────────────────────────────────")
        print(f"    │  {away_name} @ {home_name}{b2b_str}")
        print(f"    │  Winner:  {winner} ({conf:.1%})")
        print(f"    │  Score:   {away_name} {ascore} - {home_name} {hscore}")
        print(f"    │  OT:      {ot_prob:.1%}")
        print(f"    │  HomeIce: {team_pred['homeice_diff']:+.3f}  "
              f"xGF: {team_pred['xGF_diff']:+.2f}  "
              f"GSAX: {team_pred['GSAX_diff']:+.3f}")
        print(f"    │  Proj goals: {home_name} {home_proj:.2f} | "
              f"{away_name} {away_proj:.2f}")
        print(f"    ├─ Top Players ────────────────────────────")
        for p in (home_players[:5] + away_players[:5]):
            side = "H" if any(
                pp["player_id"] == p["player_id"] for pp in home_players
            ) else "A"
            print(f"    │  [{side}] {p['player_name']:24} "
                  f"G:{p['goal_prob']:.0%} "
                  f"A:{p['assist_prob']:.0%} "
                  f"P:{p['point_prob']:.0%}")
        print(f"    └──────────────────────────────────────────")

        # Write to Predictions table
        winner_team_id = home_id if team_pred["winner_is_home"] else away_id
        con.execute("""
            INSERT INTO Predictions (
                PredictionID, GameID, HomeTeamID, AwayTeamID,
                PredictedWinnerTeamID, PredictedWinner,
                PredictedHomeScore, PredictedAwayScore,
                ModelConfidencePct, PredictedOT,
                HomeIceDifferential, xGFDifferential, GSAXDifferential,
                HomeProjectedGoals, AwayProjectedGoals,
                HomeIsBackToBack, AwayIsBackToBack,
                PredictionDate, Season
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [
            pred_id, game_id, home_id, away_id,
            winner_team_id, winner,
            hscore, ascore,
            round(conf * 100, 2),
            team_pred["is_ot"],
            team_pred["homeice_diff"],
            team_pred["xGF_diff"],
            team_pred["GSAX_diff"],
            home_proj, away_proj,
            home_b2b, away_b2b,
            datetime.now(),
            SEASON,
        ])

        # Write player predictions
        pp_id = con.execute(
            "SELECT COALESCE(MAX(PlayerPredictionID), 0) FROM PlayerPredictions"
        ).fetchone()[0] + 1

        for p in home_players + away_players:
            is_home_player = any(
                pp["player_id"] == p["player_id"] for pp in home_players
            )
            con.execute("""
                INSERT INTO PlayerPredictions (
                    PlayerPredictionID, PredictionID, PlayerID, GameID,
                    GoalProbability, AssistProbability, PointProbability,
                    IsHome, PredictionDate
                ) VALUES (?,?,?,?,?,?,?,?,?)
            """, [
                pp_id, pred_id,
                p["player_id"], game_id,
                p["goal_prob"], p["assist_prob"], p["point_prob"],
                is_home_player,
                datetime.now().date(),
            ])
            pp_id += 1

        pred_id += 1
        predictions_made += 1

    print(f"\n  Predictions made: {predictions_made}")

# ── Step 4: Accuracy summary ──────────────────────────────────
print("\n" + "-"*50)
print("STEP 4: ACCURACY SUMMARY")
print("-"*50)

stats = con.execute("""
    SELECT
        COUNT(*)                                              AS total,
        SUM(CASE WHEN ActualWinner IS NOT NULL THEN 1 ELSE 0 END) AS completed,
        SUM(CASE WHEN MLCorrect = 1 THEN 1 ELSE 0 END)       AS correct,
        SUM(CASE WHEN MLCorrect = 0
             AND ActualWinner IS NOT NULL THEN 1 ELSE 0 END)  AS wrong
    FROM Predictions
    WHERE Season = '2025-26'
      AND PredictedWinner IS NOT NULL
""").fetchone()

total, completed, correct, wrong = stats
print(f"  Season predictions: {total}")
print(f"  Completed:          {completed}")
if completed and completed > 0:
    print(f"  Correct:            {correct}")
    print(f"  Wrong:              {wrong}")
    print(f"  Accuracy:           {correct/completed*100:.1f}%")
    print(f"  Model CV accuracy:  {team_model.cv_mean:.1%} ±{team_model.cv_std:.1%}")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

con.close()
