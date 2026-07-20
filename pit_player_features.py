"""
pit_player_features.py
-----------------------
Point-in-time (leak-free) per-game player features for the goal/assist/point
prop model. Parallels pit_features.py but at the skater-game grain.

Why this exists:
  The old player_model was doubly broken. (1) Its target was the player's
  SEASON average (Goals/GP) while its features were the same player's season
  per-60 rates (Goals_Per60, Points_Per60) — the same quantity rescaled, so
  it predicted a number from itself. (2) It never used per-game data at all;
  it fabricated rows by sampling random opponents around one season snapshot.

  This module instead builds, for every real skater-game in PlayerGameLog,
  features from ONLY that player's earlier games (plus the opponent's
  as-of-date goals-against), and binary targets from the actual game:
      y_goal   = 1 if the player scored >=1 goal
      y_assist = 1 if >=1 assist
      y_point  = 1 if >=1 point
  So the model learns genuine per-game probabilities with no leakage.

Cold start: early-season rates are shrunk toward the player's PRIOR-season
rate (fully known before the season, so leak-free) with weight PRIOR_W games;
rookies/priorless players fall back to league base rates.

Public API:
  load_player_games(con)            -> chronological skater-game DataFrame
  build_opp_defense(con)            -> {(team_id, game_id): opp GA/game as-of}
  build_player_table(games, opp_def, prior_rates) -> (X, meta)
  prior_season_rates(games)         -> {season: {player_id: rate dict}}
  compute_live_player_state(games)  -> final per-player accumulators (for today)
  snapshot_player(...)              -> single feature dict (shared train/serve)

Run directly for a sanity check:
    python pit_player_features.py
"""

from collections import deque
import numpy as np
import pandas as pd

ROLL_WINDOW = 10
PRIOR_W     = 8.0        # shrinkage strength (games) toward prior-season rate

# League base per-game rates (fallback when no prior season)
LEAGUE_BASE = {
    "g_pg": 0.15, "a_pg": 0.24, "p_pg": 0.34,
    "shots_pg": 1.9, "toi_min": 14.0, "sh_pct": 0.09,
}


def load_player_games(con):
    """All skater game-logs (both seasons), each tagged with its opponent
    team and back-to-back flag, in strict chronological order."""
    df = con.execute("""
        SELECT
            pl.PlayerID,
            pl.TeamID,
            CASE WHEN g.HomeTeamID = pl.TeamID THEN g.AwayTeamID
                 ELSE g.HomeTeamID END                      AS OppTeamID,
            pl.GameID,
            pl.GameDate,
            pl.Season,
            pl.IsHome,
            CASE WHEN pl.IsHome THEN COALESCE(g.HomeIsBackToBack, FALSE)
                 ELSE COALESCE(g.AwayIsBackToBack, FALSE) END AS IsB2B,
            CASE WHEN pl.GameType = 'Playoffs' THEN 1 ELSE 0 END AS IsPlayoff,
            pl.Goals,
            pl.Assists,
            pl.Points,
            COALESCE(pl.Shots, 0) AS Shots,
            COALESCE(pl.TOI, 0)   AS TOI
        FROM PlayerGameLog pl
        JOIN Players p ON pl.PlayerID = p.PlayerID
        JOIN Games   g ON pl.GameID   = g.GameID
        WHERE p.Position != 'G'
          AND pl.TOI > 0
          AND pl.Season IN ('2024-25', '2025-26')
        ORDER BY pl.GameDate ASC, pl.GameID ASC, pl.PlayerID ASC
    """).df()
    df["GameDate"] = pd.to_datetime(df["GameDate"])
    return df


def build_opp_defense(con):
    """
    {(team_id, game_id): team's season-to-date goals-against per game BEFORE
    that game}. Used to describe the opponent a player faces, point-in-time.
    """
    rows = con.execute("""
        SELECT GameID, Season, GameDate, HomeTeamID AS TeamID, AwayScore AS GA
        FROM Games WHERE HomeScore IS NOT NULL AND Season IN ('2024-25','2025-26')
        UNION ALL
        SELECT GameID, Season, GameDate, AwayTeamID, HomeScore
        FROM Games WHERE HomeScore IS NOT NULL AND Season IN ('2024-25','2025-26')
        ORDER BY GameDate ASC, GameID ASC
    """).df()
    out = {}
    acc = {}   # (season, team) -> [gp, ga]
    for r in rows.itertuples(index=False):
        key = (r.Season, int(r.TeamID))
        gp, ga = acc.get(key, (0, 0))
        out[(int(r.TeamID), int(r.GameID))] = (ga / gp) if gp else 3.0
        acc[key] = (gp + 1, ga + int(r.GA))
    return out


def prior_season_rates(games_df):
    """{season: {player_id: rate dict}} from the PREVIOUS season's totals."""
    agg = (games_df.groupby(["Season", "PlayerID"])
           .agg(gp=("GameID", "count"), g=("Goals", "sum"),
                a=("Assists", "sum"), p=("Points", "sum"),
                sh=("Shots", "sum"), toi=("TOI", "sum"))
           .reset_index())
    per_season = {}
    for season, grp in agg.groupby("Season"):
        d = {}
        for r in grp.itertuples(index=False):
            if r.gp <= 0:
                continue
            d[int(r.PlayerID)] = {
                "g_pg": r.g / r.gp, "a_pg": r.a / r.gp, "p_pg": r.p / r.gp,
                "shots_pg": r.sh / r.gp, "toi_min": (r.toi / r.gp) / 60.0,
                "sh_pct": (r.g / r.sh) if r.sh else LEAGUE_BASE["sh_pct"],
            }
        per_season[season] = d
    # Map each season to the PRIOR season's dict
    order = ["2024-25", "2025-26"]
    prior = {}
    for i, s in enumerate(order):
        prior[s] = per_season.get(order[i - 1]) if i > 0 else {}
    return prior


class _PlayerAcc:
    __slots__ = ("gp", "g", "a", "p", "sh", "toi",
                 "rg", "ra", "rp", "rsh")

    def __init__(self):
        self.gp = 0
        self.g = self.a = self.p = self.sh = self.toi = 0
        self.rg  = deque(maxlen=ROLL_WINDOW)
        self.ra  = deque(maxlen=ROLL_WINDOW)
        self.rp  = deque(maxlen=ROLL_WINDOW)
        self.rsh = deque(maxlen=ROLL_WINDOW)

    def _shrunk(self, total, prior_rate):
        return (total + PRIOR_W * prior_rate) / (self.gp + PRIOR_W)

    def update(self, g, a, p, sh, toi):
        self.gp += 1
        self.g += g; self.a += a; self.p += p; self.sh += sh; self.toi += toi
        self.rg.append(g); self.ra.append(a); self.rp.append(p); self.rsh.append(sh)


def snapshot_player(acc, prior, opp_ga_pg, is_home, is_b2b, is_playoff):
    """Single feature dict — shared by training and live prediction."""
    pr = prior if prior else LEAGUE_BASE
    g_pg  = acc._shrunk(acc.g,  pr["g_pg"])
    a_pg  = acc._shrunk(acc.a,  pr["a_pg"])
    p_pg  = acc._shrunk(acc.p,  pr["p_pg"])
    sh_pg = acc._shrunk(acc.sh, pr["shots_pg"])
    toi_min = acc._shrunk(acc.toi / 60.0, pr["toi_min"])
    sh_pct = (acc.g + PRIOR_W * pr["sh_pct"]) / (acc.sh + PRIOR_W) if (acc.sh + PRIOR_W) else pr["sh_pct"]

    def rmean(dq, fallback):
        return float(np.mean(dq)) if dq else fallback

    return {
        "g_pg":            g_pg,
        "a_pg":            a_pg,
        "p_pg":            p_pg,
        "shots_pg":        sh_pg,
        "toi_min":         toi_min,
        "sh_pct":          sh_pct,
        "roll10_g_pg":     rmean(acc.rg,  g_pg),
        "roll10_a_pg":     rmean(acc.ra,  a_pg),
        "roll10_p_pg":     rmean(acc.rp,  p_pg),
        "roll10_shots_pg": rmean(acc.rsh, sh_pg),
        "gp":              acc.gp,
        "opp_ga_pg":       opp_ga_pg,
        "is_home":         1 if is_home else 0,
        "is_b2b":          1 if is_b2b else 0,
        "is_playoff":      int(is_playoff),
    }


def build_player_table(games_df, opp_def, prior_rates):
    """Chronological pass: snapshot leak-free features BEFORE each game, then
    update the player's accumulators with the actual result."""
    accs = {}                    # (season, player) -> _PlayerAcc
    rows, meta_rows = [], []

    for r in games_df.itertuples(index=False):
        season = r.Season
        pid    = int(r.PlayerID)
        acc = accs.setdefault((season, pid), _PlayerAcc())
        prior = prior_rates.get(season, {}).get(pid)
        opp_ga = opp_def.get((int(r.OppTeamID), int(r.GameID)), 3.0)

        feats = snapshot_player(acc, prior, opp_ga,
                                bool(r.IsHome), bool(r.IsB2B), r.IsPlayoff)
        rows.append(feats)
        meta_rows.append({
            "PlayerID": pid, "GameID": int(r.GameID), "GameDate": r.GameDate,
            "Season": season, "IsPlayoff": int(r.IsPlayoff),
            "y_goal":   1 if r.Goals   >= 1 else 0,
            "y_assist": 1 if r.Assists >= 1 else 0,
            "y_point":  1 if r.Points  >= 1 else 0,
        })
        acc.update(int(r.Goals), int(r.Assists), int(r.Points),
                   int(r.Shots), float(r.TOI))

    X = pd.DataFrame(rows).astype(float)
    meta = pd.DataFrame(meta_rows)
    return X, meta


def compute_live_player_state(games_df):
    """Replay all games; return {(season, player_id): _PlayerAcc} final state
    for building today's feature rows."""
    accs = {}
    for r in games_df.itertuples(index=False):
        acc = accs.setdefault((r.Season, int(r.PlayerID)), _PlayerAcc())
        acc.update(int(r.Goals), int(r.Assists), int(r.Points),
                   int(r.Shots), float(r.TOI))
    return accs


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    from db_features import get_connection
    print("Connecting...")
    con = get_connection()
    games = load_player_games(con)
    print(f"  {len(games)} skater-games "
          f"({games['GameDate'].min().date()} -> {games['GameDate'].max().date()})")
    opp_def = build_opp_defense(con)
    prior = prior_season_rates(games)
    X, meta = build_player_table(games, opp_def, prior)
    print(f"  Feature table: {X.shape[0]} x {X.shape[1]}")
    print(f"  Features: {list(X.columns)}")
    print(f"  Base rates -> goal {meta['y_goal'].mean():.3f}  "
          f"assist {meta['y_assist'].mean():.3f}  point {meta['y_point'].mean():.3f}")
    con.close()
    print("DONE")
