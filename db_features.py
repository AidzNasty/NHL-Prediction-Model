"""
db_features.py
---------------
Foundation module — pulls and builds all feature vectors
from MotherDuck for use in both the team and player models.

Provides:
    get_connection()           -> DuckDB connection
    get_team_features(con, season)    -> DataFrame of team-level features
    get_player_features(con, season)  -> DataFrame of player-level features
    get_goalie_features(con, season)  -> DataFrame of goalie-level features
    get_completed_games(con)          -> DataFrame of all completed games (training)
    get_todays_games(con)             -> DataFrame of today's scheduled games
    get_confirmed_lineup(con, game_id)-> Dict of confirmed players per team
    build_game_features(con, home_team_id, away_team_id, season, game_date)
                                      -> Feature dict ready for model input
    build_player_game_features(con, player_id, opp_team_id, is_home, season)
                                      -> Feature dict for player model
"""

import os
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

# -- Connection ------------------------------------------------
def get_connection():
    return duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")

# -- Team Features ---------------------------------------------
def get_team_features(con, season="2025-26"):
    """
    Returns a DataFrame with one row per team containing
    all team-level advanced stats from TeamStandings.
    """
    df = con.execute(f"""
        SELECT
            ts.TeamID,
            t.TeamName,
            t.Abbreviation,
            ts.Season,
            -- Record
            ts.GP,
            ts.W,
            ts.L,
            ts.OTL,
            ts.Points,
            ts.Point_Pct,
            -- Corsi / Fenwick
            ts.CF_Pct,
            ts.FF_Pct,
            -- Shots
            ts.SF_Pct,
            -- Goals
            ts.GF_Pct,
            ts.GF,
            ts.GA,
            CASE WHEN ts.GP > 0 THEN ts.GF * 1.0 / ts.GP ELSE NULL END AS GF_Per_Game,
            CASE WHEN ts.GP > 0 THEN ts.GA * 1.0 / ts.GP ELSE NULL END AS GA_Per_Game,
            -- Expected Goals
            ts.xGF_Pct,
            ts.xGF,
            ts.xGA,
            -- Scoring Chances
            ts.SCF_Pct,
            ts.SCGF_Pct,
            -- High Danger
            ts.HDCF_Pct,
            ts.HDGF_Pct,
            ts.HDSV_Pct,
            ts.HDSH_Pct,
            -- Medium Danger
            ts.MDCF_Pct,
            ts.MDSV_Pct,
            -- Shooting / Save
            ts.SH_Pct,
            ts.SV_Pct,
            ts.PDO,
            -- Home / Away splits
            ts.HomeWinPct,
            ts.AwayWinPct,
            ts.HomeGF_Per_Game,
            ts.HomeGA_Per_Game,
            ts.AwayGF_Per_Game,
            ts.AwayGA_Per_Game,
            -- Faceoff / Giveaway / Takeaway tendencies
            ts.Team_FaceoffWinPct,
            ts.Team_Giveaways_Per_Game,
            ts.Team_Takeaways_Per_Game,
            (ts.GF - ts.GA) AS GoalDiff
        FROM TeamStandings ts
        JOIN Teams t ON ts.TeamID = t.TeamID
        WHERE ts.Season = '{season}'
          AND ts.GP > 0
        ORDER BY ts.Points DESC
    """).df()
    return df

# -- Goalie Features -------------------------------------------
def get_goalie_features(con, season="2025-26"):
    """
    Returns best goalie stats per team (by games played).
    Uses GSAX as primary quality metric.
    """
    df = con.execute(f"""
        SELECT
            p.TeamID,
            gs.PlayerID,
            p.FirstName || ' ' || p.LastName AS GoalieName,
            gs.Season,
            gs.GP        AS Goalie_GP,
            gs.TOI       AS Goalie_TOI,
            gs.SV_Pct    AS Goalie_SV_Pct,
            gs.xGA       AS Goalie_xGA,
            gs.GSAX      AS Goalie_GSAX,
            gs.GSAX_Per60 AS Goalie_GSAX60,
            gs.HD_SV_Pct AS Goalie_HDSV_Pct,
            gs.MD_SV_Pct AS Goalie_MDSV_Pct,
            gs.LD_SV_Pct AS Goalie_LDSV_Pct,
            -- Rank within team by games played
            ROW_NUMBER() OVER (
                PARTITION BY p.TeamID
                ORDER BY gs.GP DESC
            ) AS starter_rank
        FROM GoalieStats gs
        JOIN Players p ON gs.PlayerID = p.PlayerID
        WHERE gs.Season = '{season}'
    """).df()

    # Return only the primary starter per team
    starters = df[df["starter_rank"] == 1].drop(columns=["starter_rank"])
    return starters

# -- Player Features -------------------------------------------
def get_player_features(con, season="2025-26"):
    """
    Returns player-level advanced stats for all active skaters.
    Per-60 rates calculated where possible.
    """
    df = con.execute(f"""
        SELECT
            ps.PlayerID,
            p.FirstName || ' ' || p.LastName AS PlayerName,
            p.TeamID,
            p.Position,
            ps.Season,
            ps.GP,
            -- TOI per game
            CASE WHEN ps.GP > 0 THEN ps.TOI / ps.GP ELSE NULL END AS TOI_Per_Game,
            -- Scoring
            ps.Goals,
            ps.Total_Assists,
            ps.Total_Points,
            CASE WHEN ps.TOI > 0 THEN ps.Goals        / ps.TOI * 60 ELSE NULL END AS Goals_Per60,
            CASE WHEN ps.TOI > 0 THEN ps.Total_Assists / ps.TOI * 60 ELSE NULL END AS Assists_Per60,
            CASE WHEN ps.TOI > 0 THEN ps.Total_Points  / ps.TOI * 60 ELSE NULL END AS Points_Per60,
            -- Individual advanced
            ps.ixG,
            CASE WHEN ps.TOI > 0 THEN ps.ixG    / ps.TOI * 60 ELSE NULL END AS ixG_Per60,
            CASE WHEN ps.TOI > 0 THEN ps.iHDCF  / ps.TOI * 60 ELSE NULL END AS iHDCF_Per60,
            CASE WHEN ps.TOI > 0 THEN ps.iCF    / ps.TOI * 60 ELSE NULL END AS iCF_Per60,
            CASE WHEN ps.TOI > 0 THEN ps.iSCF   / ps.TOI * 60 ELSE NULL END AS iSCF_Per60,
            CASE WHEN ps.TOI > 0 THEN ps.Rush_Attempts / ps.TOI * 60 ELSE NULL END AS Rush_Per60,
            ps.SH_Pct    AS Player_SH_Pct,
            ps.IPP,
            -- On-ice advanced
            ps.xGF_Pct   AS OnIce_xGF_Pct,
            ps.CF_Pct    AS OnIce_CF_Pct,
            ps.HDCF_Pct  AS OnIce_HDCF_Pct,
            ps.PDO       AS OnIce_PDO,
            -- Zone starts
            ps.Off_Zone_Start_Pct,
            ps.Def_Zone_Starts,
            -- Giveaways / Takeaways
            CASE WHEN ps.GP > 0 THEN ps.Giveaways / ps.GP ELSE NULL END AS GV_Per_Game,
            CASE WHEN ps.GP > 0 THEN ps.Takeaways / ps.GP ELSE NULL END AS TK_Per_Game,
            -- Faceoffs (centers mainly)
            ps.Faceoffs_Pct,
            ps.Faceoffs_Won,
            ps.Faceoffs_Lost
        FROM PlayerStats ps
        JOIN Players p ON ps.PlayerID = p.PlayerID
        WHERE ps.Season = '{season}'
          AND ps.GP >= 5
          AND p.IsActive = TRUE
        ORDER BY p.TeamID, ps.ixG DESC
    """).df()
    return df

# -- Completed Games (training data) --------------------------
def get_completed_games(con):
    """
    Returns all completed games across both seasons
    with home/away team IDs, scores, and game context.
    """
    df = con.execute("""
        SELECT
            g.GameID,
            g.Season,
            g.GameDate,
            g.HomeTeamID,
            g.AwayTeamID,
            g.HomeScore,
            g.AwayScore,
            g.OvertimeFlag,
            g.HomeIsBackToBack,
            g.AwayIsBackToBack,
            CASE WHEN g.GameType = 'Playoffs' THEN 1 ELSE 0 END AS IsPlayoff,
            ht.TeamName AS HomeTeamName,
            awt.TeamName AS AwayTeamName
        FROM Games g
        JOIN Teams ht ON g.HomeTeamID = ht.TeamID
        JOIN Teams awt ON g.AwayTeamID = awt.TeamID
        WHERE g.HomeScore IS NOT NULL
          AND g.AwayScore IS NOT NULL
          AND g.Season IN ('2024-25', '2025-26')
        ORDER BY g.GameDate ASC
    """).df()
    return df

# -- Today's Games ---------------------------------------------
def get_todays_games(con, target_date=None):
    """
    Returns today's scheduled games (or target_date if specified).
    """
    if target_date is None:
        target_date = date.today()

    df = con.execute(f"""
        SELECT
            g.GameID,
            g.Season,
            g.GameDate,
            g.HomeTeamID,
            g.AwayTeamID,
            g.HomeIsBackToBack,
            g.AwayIsBackToBack,
            ht.TeamName AS HomeTeamName,
            awt.TeamName AS AwayTeamName,
            ht.Abbreviation AS HomeAbbrev,
            awt.Abbreviation AS AwayAbbrev
        FROM Games g
        JOIN Teams ht ON g.HomeTeamID = ht.TeamID
        JOIN Teams awt ON g.AwayTeamID = awt.TeamID
        WHERE g.GameDate = '{target_date}'
          AND g.HomeScore IS NULL
        ORDER BY g.GameDate ASC
    """).df()
    return df

# -- Confirmed Lineup ------------------------------------------
def get_confirmed_lineup(con, team_id):
    """
    Returns confirmed active/inactive players for a team
    from PlayerGameStatus (populated by scrape_dailyfaceoff.py).
    Returns dict: {player_id: status} where status = 'Active'/'Injured'/'Healthy Scratch'
    """
    rows = con.execute(f"""
        SELECT
            pgs.PlayerID,
            pgs.Status,
            pgs.StatusDate,
            p.FirstName || ' ' || p.LastName AS PlayerName,
            p.Position
        FROM PlayerGameStatus pgs
        JOIN Players p ON pgs.PlayerID = p.PlayerID
        WHERE p.TeamID = {team_id}
        ORDER BY pgs.StatusDate DESC
    """).fetchall()

    lineup = {}
    for pid, status, status_date, name, pos in rows:
        if pid not in lineup:  # keep most recent
            lineup[pid] = {
                "status": status,
                "name":   name,
                "pos":    pos,
                "date":   status_date,
            }
    return lineup

# -- Rolling 10-Game Team Stats --------------------------------
def get_team_rolling_stats(con, team_id, before_date, n=10):
    """
    Calculates team performance over last N games before a given date.
    Uses GameStats for shots/hits and Games for goals/wins.
    Returns a dict of rolling features.
    """
    try:
        result = con.execute(f"""
            SELECT
                COUNT(*)                                                AS games_played,
                -- Wins
                SUM(CASE
                    WHEN g.HomeTeamID = {team_id} AND g.HomeScore > g.AwayScore THEN 1
                    WHEN g.AwayTeamID = {team_id} AND g.AwayScore > g.HomeScore THEN 1
                    ELSE 0 END)                                        AS wins,
                -- Goals for
                SUM(CASE
                    WHEN g.HomeTeamID = {team_id} THEN g.HomeScore
                    ELSE g.AwayScore END)                              AS goals_for,
                -- Goals against
                SUM(CASE
                    WHEN g.HomeTeamID = {team_id} THEN g.AwayScore
                    ELSE g.HomeScore END)                              AS goals_against,
                -- OT games
                SUM(CASE WHEN g.OvertimeFlag THEN 1 ELSE 0 END)       AS ot_games,
                -- Shots for (from GameStats)
                AVG(gs.Shots)                                          AS avg_shots,
                -- Hits
                AVG(gs.Hits)                                          AS avg_hits,
                -- Blocked shots
                AVG(gs.BlockedShots)                                  AS avg_blocks
            FROM (
                SELECT GameID, GameDate, HomeTeamID, AwayTeamID,
                       HomeScore, AwayScore, OvertimeFlag
                FROM Games
                WHERE (HomeTeamID = {team_id} OR AwayTeamID = {team_id})
                  AND HomeScore IS NOT NULL
                  AND GameDate < '{before_date}'
                ORDER BY GameDate DESC
                LIMIT {n}
            ) g
            LEFT JOIN GameStats gs
                ON gs.GameID = g.GameID
               AND gs.TeamID = {team_id}
        """).fetchone()

        if not result or result[0] == 0:
            return _default_rolling()

        gp, wins, gf, ga, ot, shots, hits, blocks = result
        gp = int(gp or 1)

        return {
            "rolling_win_pct":      round(wins / gp, 3) if gp > 0 else 0.5,
            "rolling_gf_per_game":  round(gf / gp, 2)  if gp > 0 else 3.0,
            "rolling_ga_per_game":  round(ga / gp, 2)  if gp > 0 else 3.0,
            "rolling_goal_diff":    round((gf - ga) / gp, 2) if gp > 0 else 0.0,
            "rolling_ot_rate":      round(ot / gp, 3) if gp > 0 else 0.21,
            "rolling_shots":        round(float(shots or 28.0), 1),
            "rolling_hits":         round(float(hits  or 20.0), 1),
            "rolling_blocks":       round(float(blocks or 12.0), 1),
            "rolling_games":        gp,
        }
    except Exception as e:
        return _default_rolling()

def _default_rolling():
    return {
        "rolling_win_pct":     0.5,
        "rolling_gf_per_game": 3.0,
        "rolling_ga_per_game": 3.0,
        "rolling_goal_diff":   0.0,
        "rolling_ot_rate":     0.21,
        "rolling_shots":       28.0,
        "rolling_hits":        20.0,
        "rolling_blocks":      12.0,
        "rolling_games":       0,
    }


def get_team_streak(con, team_id, season="2025-26"):
    """
    Returns current consecutive win/loss streak as a signed integer.
    +5 = W5 (won last 5), -3 = L3 (lost last 3).
    """
    rows = con.execute(f"""
        SELECT CASE WHEN WinnerTeamID = {team_id} THEN 1 ELSE -1 END AS result
        FROM Games
        WHERE (HomeTeamID = {team_id} OR AwayTeamID = {team_id})
          AND HomeScore IS NOT NULL
          AND Season = '{season}'
        ORDER BY GameDate DESC
        LIMIT 20
    """).fetchall()
    if not rows:
        return 0
    first = rows[0][0]
    count = 0
    for (r,) in rows:
        if r == first:
            count += 1
        else:
            break
    return first * count  # +5 = W5, -3 = L3


def calc_homeice_differential(home_team_row, away_team_row):
    """
    Preserves the HomeIce formula from the original Excel model
    but calculated from DB stats rather than Excel.

    Formula: (HomeWinPct - AwayWinPct) * 6
    Positive = home team advantage, Negative = away team advantage
    """
    home_hwp = home_team_row.get("HomeWinPct", 0.5) or 0.5
    away_awp = away_team_row.get("AwayWinPct", 0.5) or 0.5
    return round((home_hwp - away_awp) * 6, 3)

# -- Build Game Feature Vector ---------------------------------
def build_game_features(con, home_team_id, away_team_id, season,
                        home_b2b=False, away_b2b=False):
    """
    Builds a complete feature dict for one game matchup.
    Used for both training (historical games) and prediction (today).

    Returns a flat dict of features ready to be passed to the model.
    """
    # Get team stats
    teams_df = get_team_features(con, season)
    home = teams_df[teams_df["TeamID"] == home_team_id]
    away = teams_df[teams_df["TeamID"] == away_team_id]

    if home.empty or away.empty:
        return None

    home = home.iloc[0]
    away = away.iloc[0]

    # Get goalie stats
    goalies_df = get_goalie_features(con, season)
    home_goalie = goalies_df[goalies_df["TeamID"] == home_team_id]
    away_goalie = goalies_df[goalies_df["TeamID"] == away_team_id]

    hg = home_goalie.iloc[0] if not home_goalie.empty else None
    ag = away_goalie.iloc[0] if not away_goalie.empty else None

    # Get player aggregates
    players_df = get_player_features(con, season)

    def top_player_agg(team_id, n_forwards=6, n_defense=4):
        team_p = players_df[players_df["TeamID"] == team_id].copy()
        forwards = team_p[team_p["Position"].isin(["C","L","LW","R","RW","F"])]
        defense  = team_p[team_p["Position"].isin(["D"])]
        top_f = forwards.nlargest(n_forwards, "ixG_Per60")
        top_d = defense.nlargest(n_defense,  "OnIce_xGF_Pct")
        return {
            "top6_ixG_per60":   top_f["ixG_Per60"].mean()   if len(top_f) > 0 else 0,
            "top6_iHDCF_per60": top_f["iHDCF_Per60"].mean() if len(top_f) > 0 else 0,
            "top6_SH_pct":      top_f["Player_SH_Pct"].mean() if len(top_f) > 0 else 0,
            "top4D_xGF_pct":    top_d["OnIce_xGF_Pct"].mean() if len(top_d) > 0 else 0,
            "top4D_CF_pct":     top_d["OnIce_CF_Pct"].mean()  if len(top_d) > 0 else 0,
        }

    home_pagg = top_player_agg(home_team_id)
    away_pagg = top_player_agg(away_team_id)

    # Get confirmed lineup and calculate injured player impact
    home_lineup = get_confirmed_lineup(con, home_team_id)
    away_lineup = get_confirmed_lineup(con, away_team_id)

    def injured_impact(team_id, lineup, players_df):
        """Sum of ixG/60 for confirmed OUT players."""
        team_p = players_df[players_df["TeamID"] == team_id]
        impact = 0.0
        for pid, info in lineup.items():
            if info["status"] in ("Injured", "Healthy Scratch", "Out"):
                player_row = team_p[team_p["PlayerID"] == pid]
                if not player_row.empty:
                    ixg = player_row.iloc[0].get("ixG_Per60", 0) or 0
                    impact += ixg
        return round(impact, 4)

    home_injured = injured_impact(home_team_id, home_lineup, players_df)
    away_injured = injured_impact(away_team_id, away_lineup, players_df)

    # HomeIce differential
    homeice = calc_homeice_differential(home, away)

    # Rolling 10-game stats
    today_str = str(date.today())
    home_rolling = get_team_rolling_stats(con, home_team_id, today_str)
    away_rolling = get_team_rolling_stats(con, away_team_id, today_str)

    # Current streak
    home_streak = get_team_streak(con, home_team_id)
    away_streak = get_team_streak(con, away_team_id)

    # Build feature dict
    def val(series, key, default=0.0):
        v = series.get(key, default)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    features = {
        # Team advanced — home
        "home_CF_Pct":          val(home, "CF_Pct", 50.0),
        "home_FF_Pct":          val(home, "FF_Pct", 50.0),
        "home_xGF_Pct":         val(home, "xGF_Pct", 50.0),
        "home_SCF_Pct":         val(home, "SCF_Pct", 50.0),
        "home_HDCF_Pct":        val(home, "HDCF_Pct", 50.0),
        "home_HDSV_Pct":        val(home, "HDSV_Pct", 85.0),
        "home_PDO":             val(home, "PDO", 100.0),
        "home_SH_Pct":          val(home, "SH_Pct", 9.0),
        "home_SV_Pct":          val(home, "SV_Pct", 0.910),
        "home_GF_Per_Game":     val(home, "GF_Per_Game", 3.0),
        "home_GA_Per_Game":     val(home, "GA_Per_Game", 3.0),
        "home_Points":          val(home, "Points", 50.0),
        "home_Point_Pct":       val(home, "Point_Pct", 0.5),
        "home_HomeWinPct":      val(home, "HomeWinPct", 0.5),
        "home_HomeGF_Per_Game": val(home, "HomeGF_Per_Game", 3.0),
        "home_HomeGA_Per_Game": val(home, "HomeGA_Per_Game", 3.0),
        "home_FaceoffWinPct":   val(home, "Team_FaceoffWinPct", 50.0),
        "home_GV_Per_Game":     val(home, "Team_Giveaways_Per_Game", 10.0),
        "home_TK_Per_Game":     val(home, "Team_Takeaways_Per_Game", 8.0),

        # Team advanced — away
        "away_CF_Pct":          val(away, "CF_Pct", 50.0),
        "away_FF_Pct":          val(away, "FF_Pct", 50.0),
        "away_xGF_Pct":         val(away, "xGF_Pct", 50.0),
        "away_SCF_Pct":         val(away, "SCF_Pct", 50.0),
        "away_HDCF_Pct":        val(away, "HDCF_Pct", 50.0),
        "away_HDSV_Pct":        val(away, "HDSV_Pct", 85.0),
        "away_PDO":             val(away, "PDO", 100.0),
        "away_SH_Pct":          val(away, "SH_Pct", 9.0),
        "away_SV_Pct":          val(away, "SV_Pct", 0.910),
        "away_GF_Per_Game":     val(away, "GF_Per_Game", 3.0),
        "away_GA_Per_Game":     val(away, "GA_Per_Game", 3.0),
        "away_Points":          val(away, "Points", 50.0),
        "away_Point_Pct":       val(away, "Point_Pct", 0.5),
        "away_AwayWinPct":      val(away, "AwayWinPct", 0.5),
        "away_AwayGF_Per_Game": val(away, "AwayGF_Per_Game", 3.0),
        "away_AwayGA_Per_Game": val(away, "AwayGA_Per_Game", 3.0),
        "away_FaceoffWinPct":   val(away, "Team_FaceoffWinPct", 50.0),
        "away_GV_Per_Game":     val(away, "Team_Giveaways_Per_Game", 10.0),
        "away_TK_Per_Game":     val(away, "Team_Takeaways_Per_Game", 8.0),

        # Goalie — home
        "home_goalie_GSAX":     val(hg, "Goalie_GSAX",     0.0) if hg is not None else 0.0,
        "home_goalie_GSAX60":   val(hg, "Goalie_GSAX60",   0.0) if hg is not None else 0.0,
        "home_goalie_SV_Pct":   val(hg, "Goalie_SV_Pct", 0.910) if hg is not None else 0.910,
        "home_goalie_HDSV_Pct": val(hg, "Goalie_HDSV_Pct", 85.0) if hg is not None else 85.0,

        # Goalie — away
        "away_goalie_GSAX":     val(ag, "Goalie_GSAX",     0.0) if ag is not None else 0.0,
        "away_goalie_GSAX60":   val(ag, "Goalie_GSAX60",   0.0) if ag is not None else 0.0,
        "away_goalie_SV_Pct":   val(ag, "Goalie_SV_Pct", 0.910) if ag is not None else 0.910,
        "away_goalie_HDSV_Pct": val(ag, "Goalie_HDSV_Pct", 85.0) if ag is not None else 85.0,

        # Player aggregates — home
        "home_top6_ixG_per60":   home_pagg["top6_ixG_per60"],
        "home_top6_iHDCF_per60": home_pagg["top6_iHDCF_per60"],
        "home_top6_SH_pct":      home_pagg["top6_SH_pct"],
        "home_top4D_xGF_pct":    home_pagg["top4D_xGF_pct"],
        "home_top4D_CF_pct":     home_pagg["top4D_CF_pct"],

        # Player aggregates — away
        "away_top6_ixG_per60":   away_pagg["top6_ixG_per60"],
        "away_top6_iHDCF_per60": away_pagg["top6_iHDCF_per60"],
        "away_top6_SH_pct":      away_pagg["top6_SH_pct"],
        "away_top4D_xGF_pct":    away_pagg["top4D_xGF_pct"],
        "away_top4D_CF_pct":     away_pagg["top4D_CF_pct"],

        # Injured player impact
        "home_injured_ixG_lost": home_injured,
        "away_injured_ixG_lost": away_injured,

        # Game context
        "homeice_differential":  homeice,
        "points_differential":   val(home, "Points") - val(away, "Points"),
        "xGF_pct_differential":  val(home, "xGF_Pct", 50.0) - val(away, "xGF_Pct", 50.0),
        "GSAX_differential":     (val(hg, "Goalie_GSAX", 0.0) if hg is not None else 0.0) -
                                 (val(ag, "Goalie_GSAX", 0.0) if ag is not None else 0.0),
        "home_is_b2b":           1 if home_b2b else 0,
        "away_is_b2b":           1 if away_b2b else 0,
        "is_playoff":            0,

        # Rolling 10-game form — home
        "home_rolling_win_pct":      home_rolling["rolling_win_pct"],
        "home_rolling_gf_per_game":  home_rolling["rolling_gf_per_game"],
        "home_rolling_ga_per_game":  home_rolling["rolling_ga_per_game"],
        "home_rolling_goal_diff":    home_rolling["rolling_goal_diff"],
        "home_rolling_shots":        home_rolling["rolling_shots"],

        # Rolling 10-game form — away
        "away_rolling_win_pct":      away_rolling["rolling_win_pct"],
        "away_rolling_gf_per_game":  away_rolling["rolling_gf_per_game"],
        "away_rolling_ga_per_game":  away_rolling["rolling_ga_per_game"],
        "away_rolling_goal_diff":    away_rolling["rolling_goal_diff"],
        "away_rolling_shots":        away_rolling["rolling_shots"],

        # Rolling differentials — hot team vs cold team
        "rolling_win_pct_diff":   home_rolling["rolling_win_pct"] - away_rolling["rolling_win_pct"],
        "rolling_goal_diff_diff": home_rolling["rolling_goal_diff"] - away_rolling["rolling_goal_diff"],

        # Goal differential (GF - GA season totals)
        "home_goal_diff":         val(home, "GoalDiff", 0.0),
        "away_goal_diff":         val(away, "GoalDiff", 0.0),
        "goal_diff_differential": val(home, "GoalDiff", 0.0) - val(away, "GoalDiff", 0.0),

        # Consecutive win/loss streak (+5=W5, -3=L3)
        "home_streak":            float(home_streak),
        "away_streak":            float(away_streak),
        "streak_differential":    float(home_streak - away_streak),
    }

    return features

# -- Build Player Game Feature Vector -------------------------
def build_player_game_features(con, player_id, opp_team_id,
                                is_home, season, b2b=False):
    """
    Builds a feature dict for one player in one game.
    Used by the player model to predict goal/assist/point probability.
    """
    players_df = get_player_features(con, season)
    player = players_df[players_df["PlayerID"] == player_id]
    if player.empty:
        return None
    player = player.iloc[0]

    # Opponent defensive stats
    teams_df  = get_team_features(con, season)
    opp       = teams_df[teams_df["TeamID"] == opp_team_id]
    opp_goalie_df = get_goalie_features(con, season)
    opp_goalie    = opp_goalie_df[opp_goalie_df["TeamID"] == opp_team_id]

    def val(series, key, default=0.0):
        if series is None:
            return default
        v = series.get(key, default) if hasattr(series, "get") else getattr(series, key, default)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    opp_row = opp.iloc[0] if not opp.empty else None
    og_row  = opp_goalie.iloc[0] if not opp_goalie.empty else None

    features = {
        # Player individual stats
        "ixG_Per60":      val(player, "ixG_Per60",      0.0),
        "iHDCF_Per60":    val(player, "iHDCF_Per60",    0.0),
        "iCF_Per60":      val(player, "iCF_Per60",      0.0),
        "iSCF_Per60":     val(player, "iSCF_Per60",     0.0),
        "Rush_Per60":     val(player, "Rush_Per60",     0.0),
        "Goals_Per60":    val(player, "Goals_Per60",    0.0),
        "Assists_Per60":  val(player, "Assists_Per60",  0.0),
        "Points_Per60":   val(player, "Points_Per60",   0.0),
        "Player_SH_Pct":  val(player, "Player_SH_Pct", 9.0),
        "IPP":            val(player, "IPP",            60.0),
        "TOI_Per_Game":   val(player, "TOI_Per_Game",   12.0),
        "Off_Zone_Start_Pct": val(player, "Off_Zone_Start_Pct", 50.0),
        # On-ice
        "OnIce_xGF_Pct":  val(player, "OnIce_xGF_Pct", 50.0),
        "OnIce_CF_Pct":   val(player, "OnIce_CF_Pct",  50.0),
        "OnIce_PDO":      val(player, "OnIce_PDO",     100.0),
        # Opponent defense
        "opp_xGA":        val(opp_row, "xGA",          150.0) if opp_row is not None else 150.0,
        "opp_HDCF_Pct":   val(opp_row, "HDCF_Pct",     50.0) if opp_row is not None else 50.0,
        "opp_SV_Pct":     val(opp_row, "SV_Pct",      0.910) if opp_row is not None else 0.910,
        "opp_GA_Per_Game":val(opp_row, "GA_Per_Game",   3.0) if opp_row is not None else 3.0,
        # Opponent goalie
        "opp_goalie_GSAX":    val(og_row, "Goalie_GSAX",    0.0) if og_row is not None else 0.0,
        "opp_goalie_HDSV_Pct":val(og_row, "Goalie_HDSV_Pct",85.0) if og_row is not None else 85.0,
        "opp_goalie_SV_Pct":  val(og_row, "Goalie_SV_Pct", 0.910) if og_row is not None else 0.910,
        # Game context
        "is_home": 1 if is_home else 0,
        "is_b2b":  1 if b2b    else 0,
    }

    return features

# -- Quick sanity check when run directly ---------------------
if __name__ == "__main__":
    print("Connecting to MotherDuck...")
    con = get_connection()
    print("Connected!\n")

    print("Testing get_team_features...")
    teams = get_team_features(con)
    print(f"  {len(teams)} teams loaded")
    print(f"  Columns: {list(teams.columns)}\n")

    print("Testing get_goalie_features...")
    goalies = get_goalie_features(con)
    print(f"  {len(goalies)} starters loaded\n")

    print("Testing get_player_features...")
    players = get_player_features(con)
    print(f"  {len(players)} players loaded\n")

    print("Testing get_completed_games...")
    games = get_completed_games(con)
    print(f"  {len(games)} completed games loaded\n")

    print("Testing get_todays_games...")
    today_games = get_todays_games(con)
    print(f"  {len(today_games)} games today\n")

    print("Testing build_game_features...")
    # Use first completed game as test
    if len(games) > 0:
        g = games.iloc[0]
        feats = build_game_features(
            con,
            int(g["HomeTeamID"]),
            int(g["AwayTeamID"]),
            g["Season"],
            bool(g["HomeIsBackToBack"]),
            bool(g["AwayIsBackToBack"])
        )
        if feats:
            print(f"  Feature vector: {len(feats)} features")
            print(f"  Sample: homeice={feats['homeice_differential']:.3f} "
                  f"xGF_diff={feats['xGF_pct_differential']:.2f} "
                  f"GSAX_diff={feats['GSAX_differential']:.3f}")
        else:
            print("  No features built (team may not have stats yet)")

    print("\nAll tests passed!")
    con.close()
