"""
web/server.py — NHL Prediction Model web app (live MotherDuck data)

A Flask backend that serves the same data as the Streamlit dashboard (app.py)
as JSON endpoints, plus the static frontend in web/static/.

Run:
    python web/server.py
then open http://127.0.0.1:8000

Reads MOTHERDUCK_TOKEN / MOTHERDUCK_DB from .env (same as app.py) or from
Streamlit-style secrets are NOT used here — this is a standalone server.
"""

import os
import math
import time
import threading
from datetime import date, datetime

import duckdb
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

# ── Config ────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
load_dotenv()  # also honour a .env in CWD

TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB = os.getenv("MOTHERDUCK_DB", "my_db")
SEASON = "2025-26"

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=None)

# ── DB access (single shared connection, guarded by a lock) ───────────────
_con = None
_con_lock = threading.Lock()


def get_con():
    global _con
    if _con is None:
        _con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
    return _con


# Tiny TTL cache so repeated page loads don't re-hit MotherDuck.
_cache = {}
_CACHE_TTL = 300  # seconds


_last_refresh = time.time()


def q(sql, ttl=_CACHE_TTL):
    """Run SQL, return a DataFrame, cached for `ttl` seconds."""
    now = time.time()
    hit = _cache.get(sql)
    if hit and now - hit[0] < ttl:
        return hit[1]
    with _con_lock:
        df = get_con().execute(sql).df()
    _cache[sql] = (now, df)
    return df


def has_actual_cols():
    try:
        with _con_lock:
            cols = [r[0].lower() for r in get_con().execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'PlayerPredictions'
            """).fetchall()]
        return "actualgoals" in cols
    except Exception:
        return False


# ── JSON helpers ──────────────────────────────────────────────────────────
def clean(v):
    """Make a value JSON-safe (NaN/NaT -> None, numpy -> python)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return str(v)[:10]
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
            if isinstance(v, float) and math.isnan(v):
                return None
            return v
    except Exception:
        pass
    if isinstance(v, float) and (math.isinf(v)):
        return None
    return v


def records(df):
    """DataFrame -> list[dict] JSON-safe."""
    if df is None or df.empty:
        return []
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def fmt_streak(n):
    if n is None:
        return "—"
    n = int(n)
    if n > 0:
        return f"W{n}"
    if n < 0:
        return f"L{abs(n)}"
    return "—"


# ── Shared: current-season streaks {TeamID: +/-N} ─────────────────────────
def get_all_streaks():
    """{TeamID: signed streak} — consecutive W/L run ending in the latest game.
    +5 = won last 5, -3 = lost last 3."""
    try:
        rows = q(f"""
            SELECT TeamID, result FROM (
                SELECT HomeTeamID AS TeamID,
                       CASE WHEN WinnerTeamID = HomeTeamID THEN 1 ELSE -1 END AS result,
                       GameDate
                FROM Games WHERE HomeScore IS NOT NULL AND Season = '{SEASON}'
                UNION ALL
                SELECT AwayTeamID,
                       CASE WHEN WinnerTeamID = AwayTeamID THEN 1 ELSE -1 END,
                       GameDate
                FROM Games WHERE AwayScore IS NOT NULL AND Season = '{SEASON}'
            ) ORDER BY TeamID, GameDate DESC
        """)
        streaks = {}
        for team_id, grp in rows.groupby("TeamID"):
            results = grp["result"].tolist()
            if not results:
                streaks[int(team_id)] = 0
                continue
            first = int(results[0])
            count = 0
            for r in results:
                if int(r) == first:
                    count += 1
                else:
                    break
            streaks[int(team_id)] = first * count
        return streaks
    except Exception:
        return {}


HAS_ACTUALS = has_actual_cols()


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════
@app.get("/api/status")
def api_status():
    try:
        info = q(f"""
            SELECT
                MAX(PredictionDate)                                       AS last_run,
                COUNT(*)                                                  AS total,
                SUM(CASE WHEN ActualWinner IS NOT NULL THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN MLCorrect = 1 THEN 1 ELSE 0 END)           AS correct
            FROM Predictions WHERE Season = '{SEASON}'
        """, ttl=60)
        row = info.iloc[0]
        completed = int(row["completed"] or 0)
        correct = int(row["correct"] or 0)
        acc = f"{correct/completed*100:.1f}%" if completed else "—"
        return jsonify({
            "last_run": clean(row["last_run"]),
            "season_accuracy": acc,
            "total": int(row["total"] or 0),
            "completed": completed,
            "refreshed_at": datetime.fromtimestamp(_last_refresh).strftime("%H:%M:%S"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/refresh")
def api_refresh():
    """Drop the query cache so the next requests re-hit MotherDuck for fresh data."""
    global _last_refresh, HAS_ACTUALS
    _cache.clear()
    HAS_ACTUALS = has_actual_cols()
    _last_refresh = time.time()
    return jsonify({"ok": True, "refreshed_at": datetime.fromtimestamp(_last_refresh).strftime("%H:%M:%S")})


@app.get("/api/today")
def api_today():
    """Today's predicted games. Falls back to the most recent prediction date
    if there are none for the literal calendar today (off-season friendly)."""
    today = str(date.today())
    day = today
    preds = _today_query(day)
    fallback = False
    if preds.empty:
        latest = q(f"""
            SELECT MAX(CAST(PredictionDate AS DATE)) AS d
            FROM Predictions WHERE Season = '{SEASON}' AND PredictedWinner IS NOT NULL
        """, ttl=300)
        if not latest.empty and latest.iloc[0]["d"] is not None:
            day = str(latest.iloc[0]["d"])[:10]
            preds = _today_query(day)
            fallback = day != today

    streaks = get_all_streaks()
    games = []
    for _, p in preds.iterrows():
        gid = int(p["GameID"])
        players = q(f"""
            SELECT
                pl.FirstName || ' ' || pl.LastName AS Player,
                pl.Position AS Pos,
                pp.GoalProbability   AS G,
                pp.AssistProbability AS A,
                pp.PointProbability  AS P,
                pp.IsHome
            FROM PlayerPredictions pp
            JOIN Players pl ON pp.PlayerID = pl.PlayerID
            WHERE pp.GameID = {gid}
              AND CAST(pp.PredictionDate AS DATE) = '{day}'
            ORDER BY pp.PointProbability DESC
            LIMIT 12
        """)
        games.append({
            "game_id": gid,
            "home": str(p["HomeTeam"]), "away": str(p["AwayTeam"]),
            "winner": str(p["PredictedWinner"]),
            "confidence": float(p["Confidence"]),
            "home_score": int(p["PredictedHomeScore"]),
            "away_score": int(p["PredictedAwayScore"]),
            "is_ot": bool(p["PredictedOT"]),
            "home_ice": float(p["HomeIce"]),
            "xgf": float(p["xGFDiff"]),
            "gsax": float(p["GSAXDiff"]),
            "home_b2b": bool(p["HomB2B"]), "away_b2b": bool(p["AwayB2B"]),
            "home_proj": float(p["HomeProj"]), "away_proj": float(p["AwayProj"]),
            "home_streak": fmt_streak(streaks.get(int(p["HomeTeamID"]), 0)),
            "away_streak": fmt_streak(streaks.get(int(p["AwayTeamID"]), 0)),
            "actual_winner": clean(p["ActualWinner"]),
            "actual_home": clean(p["ActualHomeScore"]),
            "actual_away": clean(p["ActualAwayScore"]),
            "players": [
                {"player": r["Player"], "pos": r["Pos"],
                 "g": clean(r["G"]), "a": clean(r["A"]), "p": clean(r["P"]),
                 "is_home": bool(r["IsHome"])}
                for _, r in players.iterrows()
            ],
        })
    return jsonify({"date": day, "fallback": fallback, "games": games})


def _today_query(day):
    return q(f"""
        SELECT
            p.GameID, p.HomeTeamID, p.AwayTeamID,
            ht.TeamName AS HomeTeam, awt.TeamName AS AwayTeam,
            p.PredictedWinner,
            COALESCE(p.PredictedHomeScore, 0) AS PredictedHomeScore,
            COALESCE(p.PredictedAwayScore, 0) AS PredictedAwayScore,
            COALESCE(p.ModelConfidencePct, 50) AS Confidence,
            p.PredictedOT,
            COALESCE(p.HomeIceDifferential, 0) AS HomeIce,
            COALESCE(p.xGFDifferential, 0)     AS xGFDiff,
            COALESCE(p.GSAXDifferential, 0)    AS GSAXDiff,
            COALESCE(p.HomeProjectedGoals, 0)  AS HomeProj,
            COALESCE(p.AwayProjectedGoals, 0)  AS AwayProj,
            COALESCE(p.HomeIsBackToBack, FALSE) AS HomB2B,
            COALESCE(p.AwayIsBackToBack, FALSE) AS AwayB2B,
            p.ActualWinner, p.ActualHomeScore, p.ActualAwayScore, p.MLCorrect
        FROM Predictions p
        JOIN Teams ht  ON p.HomeTeamID = ht.TeamID
        JOIN Teams awt ON p.AwayTeamID = awt.TeamID
        WHERE CAST(p.PredictionDate AS DATE) = '{day}'
          AND p.PredictedWinner IS NOT NULL
        ORDER BY p.GameID
    """)


@app.get("/api/player-props")
def api_player_props():
    today = str(date.today())
    day = today
    players = _props_query(day)
    if players.empty:
        latest = q(f"""
            SELECT MAX(CAST(PredictionDate AS DATE)) AS d FROM PlayerPredictions
        """)
        if not latest.empty and latest.iloc[0]["d"] is not None:
            day = str(latest.iloc[0]["d"])[:10]
            players = _props_query(day)
    return jsonify({"date": day, "players": records(players)})


def _props_query(day):
    actual_cols = ", pp.ActualGoals, pp.ActualAssists, pp.ActualPoints" if HAS_ACTUALS else ""
    return q(f"""
        SELECT
            pl.FirstName || ' ' || pl.LastName AS Player,
            pl.Position AS Pos,
            t.TeamName  AS Team,
            pp.GoalProbability   AS goal_prob,
            pp.AssistProbability AS assist_prob,
            pp.PointProbability  AS point_prob,
            pp.IsHome AS is_home
            {actual_cols}
        FROM PlayerPredictions pp
        JOIN Players pl ON pp.PlayerID = pl.PlayerID
        JOIN Teams t    ON pl.TeamID   = t.TeamID
        WHERE CAST(pp.PredictionDate AS DATE) = '{day}'
        ORDER BY pp.PointProbability DESC
    """)


@app.get("/api/accuracy")
def api_accuracy():
    s = q(f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN ActualWinner IS NOT NULL THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN MLCorrect = 1 THEN 1 ELSE 0 END) AS correct,
            SUM(CASE WHEN ModelConfidencePct >= 65 AND MLCorrect = 1 THEN 1 ELSE 0 END) AS high_c,
            SUM(CASE WHEN ModelConfidencePct >= 65 AND ActualWinner IS NOT NULL THEN 1 ELSE 0 END) AS high_t,
            SUM(CASE WHEN ModelConfidencePct BETWEEN 55 AND 64.9 AND MLCorrect = 1 THEN 1 ELSE 0 END) AS med_c,
            SUM(CASE WHEN ModelConfidencePct BETWEEN 55 AND 64.9 AND ActualWinner IS NOT NULL THEN 1 ELSE 0 END) AS med_t,
            SUM(CASE WHEN ModelConfidencePct < 55 AND MLCorrect = 1 THEN 1 ELSE 0 END) AS low_c,
            SUM(CASE WHEN ModelConfidencePct < 55 AND ActualWinner IS NOT NULL THEN 1 ELSE 0 END) AS low_t
        FROM Predictions
        WHERE Season = '{SEASON}' AND PredictedWinner IS NOT NULL
    """, ttl=120)
    r = s.iloc[0]
    total = int(r["total"] or 0); completed = int(r["completed"] or 0); correct = int(r["correct"] or 0)
    high_t = int(r["high_t"] or 0); med_t = int(r["med_t"] or 0); low_t = int(r["low_t"] or 0)

    def pct(c, t):
        return round(int(c or 0) / t * 100, 1) if t else 0.0

    tiers = [
        {"tier": "High (≥65%)",    "acc": pct(r["high_c"], high_t), "n": high_t},
        {"tier": "Medium (55–65%)", "acc": pct(r["med_c"], med_t),  "n": med_t},
        {"tier": "Low (<55%)",      "acc": pct(r["low_c"], low_t),  "n": low_t},
    ]

    recent = q(f"""
        SELECT
            CAST(p.PredictionDate AS DATE) AS Date,
            awt.TeamName AS Away, ht.TeamName AS Home,
            p.PredictedWinner AS Predicted,
            ROUND(COALESCE(p.ModelConfidencePct, 0), 1) AS Conf,
            p.ActualWinner AS Actual,
            CASE WHEN p.ActualWinner IS NULL THEN 'Pending'
                 WHEN p.MLCorrect = 1 THEN 'Correct'
                 ELSE 'Wrong' END AS Result
        FROM Predictions p
        JOIN Teams ht  ON p.HomeTeamID = ht.TeamID
        JOIN Teams awt ON p.AwayTeamID = awt.TeamID
        WHERE p.Season = '{SEASON}' AND p.PredictedWinner IS NOT NULL
        ORDER BY p.PredictionDate DESC
        LIMIT 25
    """, ttl=120)

    rolling = []
    if completed >= 5:
        rdf = q(f"""
            SELECT CAST(PredictionDate AS DATE) AS Date, MLCorrect
            FROM Predictions
            WHERE Season = '{SEASON}' AND ActualWinner IS NOT NULL AND MLCorrect IS NOT NULL
            ORDER BY PredictionDate
        """, ttl=120)
        if len(rdf) >= 5:
            rdf["Rolling"] = rdf["MLCorrect"].rolling(10, min_periods=1).mean() * 100
            rolling = [{"date": clean(d), "rolling": round(float(v), 1)}
                       for d, v in zip(rdf["Date"], rdf["Rolling"])]

    return jsonify({
        "accuracy": pct(correct, completed),
        "correct": correct, "completed": completed, "total": total,
        "cv": "60.5% ±1.8%",
        "tiers": tiers,
        "recent": records(recent),
        "rolling": rolling,
    })


@app.get("/api/team-stats")
def api_team_stats():
    season = request.args.get("season", SEASON)
    if season not in (SEASON, "2024-25"):
        season = SEASON
    teams = q(f"""
        WITH game_goals AS (
            SELECT TeamID, SUM(GF) AS total_GF, SUM(GA) AS total_GA, COUNT(*) AS total_GP
            FROM (
                SELECT HomeTeamID AS TeamID, HomeScore AS GF, AwayScore AS GA
                FROM Games WHERE Season = '{season}' AND HomeScore IS NOT NULL
                UNION ALL
                SELECT AwayTeamID AS TeamID, AwayScore AS GF, HomeScore AS GA
                FROM Games WHERE Season = '{season}' AND AwayScore IS NOT NULL
            ) GROUP BY TeamID
        )
        SELECT
            t.TeamName AS Team,
            ts.GP, ts.W, ts.L, ts.OTL, ts.Points,
            ROUND(ts.Point_Pct * 100, 1) AS "Point%",
            ROUND(ts.CF_Pct, 1)  AS "CF%",
            ROUND(ts.FF_Pct, 1)  AS "FF%",
            ROUND(ts.xGF_Pct, 1) AS "xGF%",
            ROUND(ts.HDCF_Pct, 1) AS "HDCF%",
            ROUND(ts.PDO, 1) AS "PDO",
            ROUND(ts.SV_Pct, 2) AS "SV%",
            ROUND(ts.SH_Pct, 1) AS "SH%",
            ROUND(gg.total_GF * 1.0 / NULLIF(gg.total_GP, 0), 2) AS "GF/G",
            ROUND(gg.total_GA * 1.0 / NULLIF(gg.total_GP, 0), 2) AS "GA/G",
            ROUND(ts.HomeWinPct * 100, 1) AS "Home W%",
            ROUND(ts.AwayWinPct * 100, 1) AS "Away W%",
            ROUND(ts.Team_FaceoffWinPct, 1) AS "FO%",
            (ts.GF - ts.GA) AS "Goal Diff"
        FROM TeamStandings ts
        JOIN Teams t ON ts.TeamID = t.TeamID
        LEFT JOIN game_goals gg ON ts.TeamID = gg.TeamID
        WHERE ts.Season = '{season}' AND ts.GP > 0
        ORDER BY ts.Points DESC
    """, ttl=300)
    return jsonify({"season": season, "teams": records(teams)})


@app.get("/api/streaks/teams")
def api_streaks_teams():
    all_streaks = get_all_streaks()
    teams = q(f"""
        WITH team_games AS (
            SELECT TeamID, GameDate, WinnerTeamID, OvertimeFlag,
                CASE WHEN TeamID = HomeTeamID THEN HomeScore ELSE AwayScore END AS GF,
                CASE WHEN TeamID = HomeTeamID THEN AwayScore ELSE HomeScore END AS GA,
                ROW_NUMBER() OVER (PARTITION BY TeamID ORDER BY GameDate DESC) AS rn
            FROM (
                SELECT HomeTeamID AS TeamID, GameDate, HomeScore, AwayScore, WinnerTeamID, OvertimeFlag, HomeTeamID, AwayTeamID
                FROM Games WHERE Season = '{SEASON}' AND HomeScore IS NOT NULL
                UNION ALL
                SELECT AwayTeamID AS TeamID, GameDate, HomeScore, AwayScore, WinnerTeamID, OvertimeFlag, HomeTeamID, AwayTeamID
                FROM Games WHERE Season = '{SEASON}' AND AwayScore IS NOT NULL
            )
        ),
        last10 AS (
            SELECT TeamID, COUNT(*) AS GP10,
                SUM(CASE WHEN WinnerTeamID = TeamID THEN 1 ELSE 0 END) AS W10,
                SUM(CASE WHEN WinnerTeamID != TeamID AND OvertimeFlag = TRUE THEN 1 ELSE 0 END) AS OTL10,
                SUM(CASE WHEN WinnerTeamID != TeamID AND (OvertimeFlag = FALSE OR OvertimeFlag IS NULL) THEN 1 ELSE 0 END) AS L10,
                ROUND(AVG(GF), 2) AS GF_avg, ROUND(AVG(GA), 2) AS GA_avg, MAX(GameDate) AS LastGame
            FROM team_games WHERE rn <= 10 GROUP BY TeamID
        )
        SELECT t.TeamID, t.TeamName AS Team, l.W10, l.OTL10, l.L10,
            l.GF_avg AS "GF/G (L10)", l.GA_avg AS "GA/G (L10)",
            (ts.GF - ts.GA) AS "Goal Diff", CAST(l.LastGame AS DATE) AS "Last Game"
        FROM last10 l
        JOIN Teams t ON l.TeamID = t.TeamID
        JOIN TeamStandings ts ON ts.TeamID = t.TeamID AND ts.Season = '{SEASON}'
        ORDER BY l.W10 DESC, l.OTL10 DESC
    """, ttl=300)
    out = []
    for _, r in teams.iterrows():
        w10 = int(r["W10"] or 0)
        tier = ("On Fire" if w10 >= 8 else "Hot" if w10 >= 6 else
                "Average" if w10 == 5 else "Cool" if w10 == 4 else "Cold")
        out.append({
            "team": r["Team"],
            "streak": fmt_streak(all_streaks.get(int(r["TeamID"]), 0)),
            "form": tier,
            "record": f"{w10}-{int(r['L10'] or 0)}-{int(r['OTL10'] or 0)}",
            "w10": w10,
            "goal_diff": clean(r["Goal Diff"]),
            "gf_l10": clean(r["GF/G (L10)"]),
            "ga_l10": clean(r["GA/G (L10)"]),
            "last_game": clean(r["Last Game"]),
        })
    return jsonify({"teams": out})


@app.get("/api/streaks/players")
def api_streaks_players():
    pos = request.args.get("pos", "All Skaters")
    team = request.args.get("team", "All Teams")

    if pos == "Forwards":
        pos_clause = "AND pl.Position IN ('C','LW','RW','F')"
    elif pos == "Defense":
        pos_clause = "AND pl.Position IN ('D')"
    elif pos == "Goalies":
        pos_clause = "AND pl.Position IN ('G')"
    else:
        pos_clause = "AND pl.Position NOT IN ('G')"

    team_clause = ""
    if team and team != "All Teams":
        safe = team.replace("'", "''")
        team_clause = f"AND t.TeamName = '{safe}'"

    players = q(f"""
        WITH ranked AS (
            SELECT gl.PlayerID, gl.Goals, gl.Assists, gl.Points, gl.PlusMinus, gl.Shots, gl.TOI,
                ROW_NUMBER() OVER (PARTITION BY gl.PlayerID ORDER BY gl.GameDate DESC) AS rn
            FROM PlayerGameLog gl
            JOIN Players pl ON gl.PlayerID = pl.PlayerID
            JOIN Teams t ON pl.TeamID = t.TeamID
            WHERE gl.Season = '{SEASON}' {pos_clause} {team_clause}
        )
        SELECT pl.FirstName || ' ' || pl.LastName AS Player, pl.Position AS Pos, t.TeamName AS Team,
            COUNT(*) AS "GP", SUM(r.Goals) AS G, SUM(r.Assists) AS A, SUM(r.Points) AS PTS,
            ROUND(SUM(r.Points) * 1.0 / COUNT(*), 2) AS "Pts/G",
            SUM(r.PlusMinus) AS "PlusMinus", SUM(r.Shots) AS SOG
        FROM ranked r
        JOIN Players pl ON r.PlayerID = pl.PlayerID
        JOIN Teams t ON pl.TeamID = t.TeamID
        WHERE r.rn <= 10 {pos_clause} {team_clause}
        GROUP BY pl.PlayerID, pl.FirstName, pl.LastName, pl.Position, t.TeamName
        HAVING COUNT(*) >= 5
        ORDER BY SUM(r.Points) DESC
        LIMIT 100
    """, ttl=300)
    out = []
    for _, r in players.iterrows():
        pts = int(r["PTS"] or 0)
        tier = ("On Fire" if pts >= 12 else "Hot" if pts >= 8 else
                "Average" if pts >= 5 else "Cool" if pts >= 3 else "Cold")
        out.append({
            "player": r["Player"], "pos": r["Pos"], "team": r["Team"], "form": tier,
            "gp": int(r["GP"] or 0), "g": int(r["G"] or 0), "a": int(r["A"] or 0),
            "pts": pts, "ptspg": clean(r["Pts/G"]),
            "plusminus": clean(r["PlusMinus"]), "sog": int(r["SOG"] or 0),
        })
    teams = q("SELECT TeamName FROM Teams ORDER BY TeamName")
    return jsonify({"players": out, "teams": [r["TeamName"] for _, r in teams.iterrows()]})


def series_win_prob(wins_a, wins_b, game_num, p_home, p_away):
    if wins_a >= 4:
        return 1.0
    if wins_b >= 4:
        return 0.0
    a_at_home = game_num in (1, 2, 5, 7)
    p = p_home if a_at_home else p_away
    return (p * series_win_prob(wins_a + 1, wins_b, game_num + 1, p_home, p_away) +
            (1 - p) * series_win_prob(wins_a, wins_b + 1, game_num + 1, p_home, p_away))


@app.get("/api/playoffs")
def api_playoffs():
    playoff_df = q(f"""
        SELECT g.GameID, g.GameDate,
            t1.TeamName AS Away, t1.TeamID AS AwayID, t1.Abbreviation AS AwayAbbr,
            t2.TeamName AS Home, t2.TeamID AS HomeID, t2.Abbreviation AS HomeAbbr,
            g.AwayScore, g.HomeScore,
            t3.TeamName AS Winner, t3.TeamID AS WinnerID, g.OvertimeFlag,
            LEAST(t1.TeamID, t2.TeamID) AS T1ID, GREATEST(t1.TeamID, t2.TeamID) AS T2ID
        FROM Games g
        JOIN Teams t1 ON g.AwayTeamID = t1.TeamID
        JOIN Teams t2 ON g.HomeTeamID = t2.TeamID
        LEFT JOIN Teams t3 ON g.WinnerTeamID = t3.TeamID
        WHERE g.GameType = 'Playoffs' AND g.Season = '{SEASON}'
        ORDER BY g.GameDate
    """, ttl=300)
    strength_df = q(f"""
        SELECT ts.TeamID, t.TeamName, t.Conference, t.Abbreviation,
            ts.Point_Pct, ts.xGF_Pct, ts.GF, ts.GA, ts.GP
        FROM TeamStandings ts JOIN Teams t ON ts.TeamID = t.TeamID
        WHERE ts.Season = '{SEASON}'
    """, ttl=300)

    if playoff_df.empty:
        return jsonify({"series": [], "east": [], "west": []})

    strength = {}
    for _, row in strength_df.iterrows():
        strength[int(row["TeamID"])] = {
            "name": row["TeamName"], "abbr": row["Abbreviation"],
            "conf": row["Conference"],
            "pt_pct": float(row["Point_Pct"] or 0.5),
            "xgf_pct": float(row["xGF_Pct"] or 50) / 100,
        }

    series_map = {}
    for _, game in playoff_df.iterrows():
        key = (int(game["T1ID"]), int(game["T2ID"]))
        series_map.setdefault(key, []).append(game)

    series_list = []
    for key, games in series_map.items():
        gs = sorted(games, key=lambda g: g["GameDate"])
        g1 = gs[0]
        home_ice_id = int(g1["HomeID"]); away_seed_id = int(g1["AwayID"])
        hi_wins = sum(1 for g in gs if pd.notna(g["WinnerID"]) and int(g["WinnerID"]) == home_ice_id)
        opp_wins = sum(1 for g in gs if pd.notna(g["WinnerID"]) and int(g["WinnerID"]) == away_seed_id)
        played = [g for g in gs if pd.notna(g["HomeScore"])]
        next_game_num = len(played) + 1

        hi_s = strength.get(home_ice_id, {"pt_pct": 0.5, "xgf_pct": 0.5})
        opp_s = strength.get(away_seed_id, {"pt_pct": 0.5, "xgf_pct": 0.5})
        HOME_ICE_BONUS = 0.04
        adj = (hi_s["pt_pct"] - opp_s["pt_pct"]) * 0.25 + (hi_s["xgf_pct"] - opp_s["xgf_pct"]) * 0.15
        adj = max(-0.20, min(0.20, adj))
        p_home = min(0.82, max(0.18, 0.5 + adj + HOME_ICE_BONUS))
        p_away = min(0.82, max(0.18, 0.5 + adj - HOME_ICE_BONUS))
        wp = (series_win_prob(hi_wins, opp_wins, next_game_num, p_home, p_away)
              if hi_wins < 4 and opp_wins < 4 else (1.0 if hi_wins == 4 else 0.0))

        game_log = []
        for i, g in enumerate(gs, 1):
            if pd.notna(g["HomeScore"]):
                ot = " (OT)" if g["OvertimeFlag"] else ""
                game_log.append({"game": f"G{i}", "date": clean(g["GameDate"]),
                                 "away": g["Away"], "home": g["Home"],
                                 "score": f"{int(g['AwayScore'])}–{int(g['HomeScore'])}{ot}",
                                 "winner": g["Winner"] or "—"})
            else:
                game_log.append({"game": f"G{i}", "date": clean(g["GameDate"]),
                                 "away": g["Away"], "home": g["Home"],
                                 "score": "—", "winner": "Pending"})

        series_list.append({
            "home_ice_id": home_ice_id, "home_ice_name": g1["Home"],
            "opp_id": away_seed_id, "opp_name": g1["Away"],
            "hi_wins": hi_wins, "opp_wins": opp_wins,
            "win_prob_hi": round(wp, 4), "win_prob_opp": round(1 - wp, 4),
            "series_over": hi_wins == 4 or opp_wins == 4,
            "conf": strength.get(home_ice_id, {}).get("conf", "Unknown"),
            "games": game_log,
        })

    series_list.sort(key=lambda s: (s["series_over"], -abs(s["win_prob_hi"] - 0.5)))

    east, west = [], []
    for s in series_list:
        pred_winner = s["home_ice_name"] if s["win_prob_hi"] >= 0.5 else s["opp_name"]
        pred_prob = max(s["win_prob_hi"], s["win_prob_opp"])
        row = {"matchup": f"{s['home_ice_name']} vs {s['opp_name']}",
               "winner": pred_winner, "confidence": f"{int(pred_prob*100)}% confident",
               "series": f"{s['hi_wins']}–{s['opp_wins']}"}
        (east if s["conf"] == "Eastern" else west).append(row)

    return jsonify({"series": series_list, "east": east, "west": west})


# ── Static frontend ───────────────────────────────────────────────────────
@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("MOTHERDUCK_TOKEN not set — add it to .env")
    port = int(os.getenv("PORT", "8000"))
    print(f"NHL Predictions web app → http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
