"""
pit_features.py
----------------
Point-in-time (leak-free) team features for NHL game prediction.

Every feature for a given game is computed using ONLY games that were
completed STRICTLY BEFORE that game's date. This eliminates the target
leakage in the old model, which described every game — October through
April — with a single end-of-season TeamStandings snapshot.

Data reality (verified against MotherDuck 2026-07):
  - TeamStandings holds only season-FINAL aggregates (no date column),
    so advanced stats (CF%, xGF%, HDCF%, PDO ...) CANNOT be made
    point-in-time and are intentionally excluded here.
  - GameStats has no advanced metrics either (box score only).
  - Games is complete and dated -> everything below is derived from it,
    optionally enriched with GameStats shots.

What it produces, per game (all as-of the morning of the game):
  - Elo rating (carried across seasons w/ regression to mean, home-ice
    adjusted, mild margin-of-victory) + Elo win probability
  - Season-to-date points%, win%, GF/GA per game, goal diff per game
  - Home/road split records
  - Rolling last-10 form (win% and goal diff)
  - Rest days + back-to-back flags
  - Games-played context (sample-size / early-season signal)

Public API:
  load_games(con)                 -> chronological completed-games DataFrame
  build_feature_table(games_df)   -> (X, meta) leak-free feature table
  ELO_STATE / compute_live_elo()  -> ratings usable for TODAY's games

Run directly for a sanity check:
    python pit_features.py
"""

from collections import deque, defaultdict
import numpy as np
import pandas as pd

# ── Elo configuration ─────────────────────────────────────────
ELO_BASE          = 1500.0
ELO_K             = 6.0     # per-game update; NHL has 82 games -> keep low
ELO_HOME_ADV      = 50.0    # ~ home-ice edge in Elo points
ELO_SEASON_REGRESS= 0.30    # fraction pulled back to mean between seasons
ELO_MOV           = True    # margin-of-victory multiplier (538-style)

ROLL_WINDOW       = 10


def load_games(con):
    """
    All completed games (both seasons, regular + playoff) in strict
    chronological order. Playoffs included for Elo continuity but flagged.
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
            COALESCE(g.OvertimeFlag, FALSE)     AS OvertimeFlag,
            COALESCE(g.HomeIsBackToBack, FALSE) AS HomeIsBackToBack,
            COALESCE(g.AwayIsBackToBack, FALSE) AS AwayIsBackToBack,
            CASE WHEN g.GameType = 'Playoffs' THEN 1 ELSE 0 END AS IsPlayoff
        FROM Games g
        WHERE g.HomeScore IS NOT NULL
          AND g.AwayScore IS NOT NULL
          AND g.Season IN ('2024-25', '2025-26')
        ORDER BY g.GameDate ASC, g.GameID ASC
    """).df()
    df["GameDate"] = pd.to_datetime(df["GameDate"])
    return df


# ── Per-team running accumulators ─────────────────────────────
class _TeamAcc:
    """Season-to-date + rolling accumulators for one team."""
    __slots__ = ("gp", "wins", "otl", "gf", "ga",
                 "home_gp", "home_w", "away_gp", "away_w",
                 "roll_win", "roll_gd", "last_date")

    def __init__(self):
        self.gp = 0
        self.wins = 0      # total wins (incl. OT/SO)
        self.otl = 0       # overtime/shootout losses (worth 1 pt)
        self.gf = 0
        self.ga = 0
        self.home_gp = 0
        self.home_w = 0
        self.away_gp = 0
        self.away_w = 0
        self.roll_win = deque(maxlen=ROLL_WINDOW)
        self.roll_gd  = deque(maxlen=ROLL_WINDOW)
        self.last_date = None

    # -- feature snapshot BEFORE the current game is applied --
    def points_pct(self):
        if self.gp == 0:
            return 0.5
        return (2 * self.wins + self.otl) / (2 * self.gp)

    def win_pct(self):
        return self.wins / self.gp if self.gp else 0.5

    def gf_pg(self):
        return self.gf / self.gp if self.gp else 3.0

    def ga_pg(self):
        return self.ga / self.gp if self.gp else 3.0

    def goal_diff_pg(self):
        return (self.gf - self.ga) / self.gp if self.gp else 0.0

    def home_win_pct(self):
        return self.home_w / self.home_gp if self.home_gp else 0.5

    def away_win_pct(self):
        return self.away_w / self.away_gp if self.away_gp else 0.5

    def roll_win_pct(self):
        return float(np.mean(self.roll_win)) if self.roll_win else 0.5

    def roll_goal_diff(self):
        return float(np.mean(self.roll_gd)) if self.roll_gd else 0.0

    def rest_days(self, game_date):
        if self.last_date is None:
            return 3.0  # neutral season-opener rest
        d = (game_date - self.last_date).days
        return float(min(d, 14))  # cap long breaks

    # -- update AFTER recording features --
    def update(self, is_home, gf, ga, won, ot_loss, game_date):
        self.gp += 1
        self.gf += gf
        self.ga += ga
        if won:
            self.wins += 1
        if ot_loss:
            self.otl += 1
        if is_home:
            self.home_gp += 1
            if won:
                self.home_w += 1
        else:
            self.away_gp += 1
            if won:
                self.away_w += 1
        self.roll_win.append(1 if won else 0)
        self.roll_gd.append(gf - ga)
        self.last_date = game_date


# ── Elo helpers ───────────────────────────────────────────────
def _elo_expected(r_home, r_away):
    """Expected home win prob including home-ice advantage."""
    return 1.0 / (1.0 + 10 ** (-((r_home + ELO_HOME_ADV) - r_away) / 400.0))

def _mov_multiplier(goal_margin, elo_diff):
    """FiveThirtyEight-style margin-of-victory multiplier (dampened)."""
    if not ELO_MOV:
        return 1.0
    margin = max(abs(goal_margin), 1)
    return np.log(margin + 1.0) * (2.2 / ((elo_diff * 0.001) + 2.2))


# ── Shared feature snapshot ───────────────────────────────────
# Single source of truth used by BOTH training (build_feature_table)
# and live prediction (build_live_feature_row), so the feature vector
# is constructed identically in both paths.
def snapshot_features(home_acc, away_acc, r_home, r_away, game_date,
                      home_b2b, away_b2b, is_playoff):
    elo_home_prob = _elo_expected(r_home, r_away)
    return {
        # Elo
        "home_elo":            r_home,
        "away_elo":            r_away,
        "elo_diff":            r_home - r_away,
        "elo_home_prob":       elo_home_prob,
        # Season-to-date strength
        "home_pts_pct":        home_acc.points_pct(),
        "away_pts_pct":        away_acc.points_pct(),
        "pts_pct_diff":        home_acc.points_pct() - away_acc.points_pct(),
        "home_win_pct":        home_acc.win_pct(),
        "away_win_pct":        away_acc.win_pct(),
        # Scoring
        "home_gf_pg":          home_acc.gf_pg(),
        "home_ga_pg":          home_acc.ga_pg(),
        "away_gf_pg":          away_acc.gf_pg(),
        "away_ga_pg":          away_acc.ga_pg(),
        "home_goal_diff_pg":   home_acc.goal_diff_pg(),
        "away_goal_diff_pg":   away_acc.goal_diff_pg(),
        "goal_diff_pg_diff":   home_acc.goal_diff_pg() - away_acc.goal_diff_pg(),
        # Matchup-specific offense/defense edges
        "home_off_vs_away_def": home_acc.gf_pg() - away_acc.ga_pg(),
        "away_off_vs_home_def": away_acc.gf_pg() - home_acc.ga_pg(),
        # Venue splits (home team's home record vs away team's road record)
        "home_home_win_pct":   home_acc.home_win_pct(),
        "away_away_win_pct":   away_acc.away_win_pct(),
        "venue_split_diff":    home_acc.home_win_pct() - away_acc.away_win_pct(),
        # Rolling last-10 form
        "home_roll10_win_pct": home_acc.roll_win_pct(),
        "away_roll10_win_pct": away_acc.roll_win_pct(),
        "roll10_win_pct_diff": home_acc.roll_win_pct() - away_acc.roll_win_pct(),
        "home_roll10_goal_diff": home_acc.roll_goal_diff(),
        "away_roll10_goal_diff": away_acc.roll_goal_diff(),
        # Rest / fatigue
        "home_rest_days":      home_acc.rest_days(game_date),
        "away_rest_days":      away_acc.rest_days(game_date),
        "rest_diff":           home_acc.rest_days(game_date) - away_acc.rest_days(game_date),
        "home_is_b2b":         1 if home_b2b else 0,
        "away_is_b2b":         1 if away_b2b else 0,
        # Sample-size / early-season context
        "home_gp":             home_acc.gp,
        "away_gp":             away_acc.gp,
        "min_gp":              min(home_acc.gp, away_acc.gp),
        "is_playoff":          int(is_playoff),
    }


def build_feature_table(games_df):
    """
    Single chronological pass. For each game, snapshot leak-free features
    (state BEFORE the game), then update accumulators + Elo with the result.

    Returns:
        X    : DataFrame of features (one row per game, same order as input)
        meta : DataFrame with GameID, GameDate, Season, IsPlayoff,
               home/away ids, and y_home_win target
    """
    elo = defaultdict(lambda: ELO_BASE)
    accs = {}                       # (season, team_id) -> _TeamAcc
    cur_season = None

    rows = []
    meta_rows = []

    for g in games_df.itertuples(index=False):
        season   = g.Season
        home_id  = int(g.HomeTeamID)
        away_id  = int(g.AwayTeamID)
        gdate    = g.GameDate
        hs, as_  = int(g.HomeScore), int(g.AwayScore)
        ot       = bool(g.OvertimeFlag)
        home_win = hs > as_

        # -- season rollover: regress Elo toward the mean, reset records --
        if season != cur_season:
            if cur_season is not None:
                for t in list(elo.keys()):
                    elo[t] = ELO_BASE + (1 - ELO_SEASON_REGRESS) * (elo[t] - ELO_BASE)
            cur_season = season

        home_acc = accs.setdefault((season, home_id), _TeamAcc())
        away_acc = accs.setdefault((season, away_id), _TeamAcc())

        r_home = elo[home_id]
        r_away = elo[away_id]

        # ---- leak-free feature snapshot (BEFORE applying result) ----
        feats = snapshot_features(
            home_acc, away_acc, r_home, r_away, gdate,
            bool(g.HomeIsBackToBack), bool(g.AwayIsBackToBack), int(g.IsPlayoff)
        )
        elo_home_prob = feats["elo_home_prob"]
        rows.append(feats)
        meta_rows.append({
            "GameID":    int(g.GameID),
            "GameDate":  gdate,
            "Season":    season,
            "IsPlayoff": int(g.IsPlayoff),
            "HomeTeamID": home_id,
            "AwayTeamID": away_id,
            "HomeScore": hs,
            "AwayScore": as_,
            "OvertimeFlag": int(ot),
            "y_home_win": int(home_win),
        })

        # ---- update accumulators (regular season only for records) ----
        # Records reset per season; playoffs still update Elo but we keep
        # season-to-date record accumulators regular-season-focused.
        home_ot_loss = ot and not home_win
        away_ot_loss = ot and home_win
        home_acc.update(True,  hs, as_, home_win,      home_ot_loss, gdate)
        away_acc.update(False, as_, hs, (not home_win), away_ot_loss, gdate)

        # ---- update Elo ----
        actual = 1.0 if home_win else 0.0
        mult   = _mov_multiplier(hs - as_, abs(r_home - r_away))
        delta  = ELO_K * mult * (actual - elo_home_prob)
        elo[home_id] = r_home + delta
        elo[away_id] = r_away - delta

    X = pd.DataFrame(rows).astype(float)
    meta = pd.DataFrame(meta_rows)
    return X, meta


def compute_live_elo(games_df):
    """
    Returns {team_id: current_elo} after processing all completed games,
    plus the per-(season,team) accumulators, for predicting TODAY's games.
    """
    elo = defaultdict(lambda: ELO_BASE)
    accs = {}
    cur_season = None
    for g in games_df.itertuples(index=False):
        season = g.Season
        if season != cur_season:
            if cur_season is not None:
                for t in list(elo.keys()):
                    elo[t] = ELO_BASE + (1 - ELO_SEASON_REGRESS) * (elo[t] - ELO_BASE)
            cur_season = season
        home_id, away_id = int(g.HomeTeamID), int(g.AwayTeamID)
        hs, as_ = int(g.HomeScore), int(g.AwayScore)
        home_win = hs > as_
        r_home, r_away = elo[home_id], elo[away_id]
        exp = _elo_expected(r_home, r_away)
        mult = _mov_multiplier(hs - as_, abs(r_home - r_away))
        delta = ELO_K * mult * ((1.0 if home_win else 0.0) - exp)
        elo[home_id] = r_home + delta
        elo[away_id] = r_away - delta
        home_acc = accs.setdefault((season, home_id), _TeamAcc())
        away_acc = accs.setdefault((season, away_id), _TeamAcc())
        home_ot_loss = bool(g.OvertimeFlag) and not home_win
        away_ot_loss = bool(g.OvertimeFlag) and home_win
        home_acc.update(True,  hs, as_, home_win,      home_ot_loss, g.GameDate)
        away_acc.update(False, as_, hs, (not home_win), away_ot_loss, g.GameDate)
    return dict(elo), accs, cur_season


def build_live_feature_row(elo, accs, season, home_id, away_id, game_date,
                           home_b2b=False, away_b2b=False, is_playoff=False):
    """
    Build ONE feature dict for an upcoming game using the live Elo/accumulator
    state from compute_live_elo(). Uses the same snapshot_features() code path
    as training, so train/serve feature construction is identical.

    game_date may be a str or datetime; converted to Timestamp for rest-day calc.
    """
    gdate = pd.to_datetime(game_date)
    r_home = elo.get(int(home_id), ELO_BASE)
    r_away = elo.get(int(away_id), ELO_BASE)
    home_acc = accs.get((season, int(home_id)), _TeamAcc())
    away_acc = accs.get((season, int(away_id)), _TeamAcc())
    return snapshot_features(
        home_acc, away_acc, r_home, r_away, gdate,
        home_b2b, away_b2b, is_playoff
    )


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    from db_features import get_connection
    print("Connecting to MotherDuck...")
    con = get_connection()
    games = load_games(con)
    print(f"  Loaded {len(games)} completed games "
          f"({games['GameDate'].min().date()} -> {games['GameDate'].max().date()})")

    X, meta = build_feature_table(games)
    print(f"  Built feature table: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  Features: {list(X.columns)}")
    print(f"  Home win rate (target): {meta['y_home_win'].mean():.3f}")

    # Elo-only accuracy as a leak-free sanity baseline
    reg = meta[meta["IsPlayoff"] == 0]
    reg_X = X.loc[reg.index]
    elo_pick_home = (reg_X["elo_home_prob"] >= 0.5).astype(int)
    elo_acc = (elo_pick_home.values == reg["y_home_win"].values).mean()
    always_home = reg["y_home_win"].mean()
    print(f"\n  Leak-free baselines (regular season, all games in-sample state):")
    print(f"    Always-home : {always_home:.3f}")
    print(f"    Elo pick    : {elo_acc:.3f}")

    # Elo top/bottom teams at season end
    final_elo, accs, season = compute_live_elo(games)
    teams = con.execute("SELECT TeamID, TeamName FROM Teams").df().set_index("TeamID")["TeamName"].to_dict()
    ranked = sorted(final_elo.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n  Top 5 Elo (through {season}):")
    for tid, r in ranked[:5]:
        print(f"    {teams.get(tid, tid):24} {r:.0f}")
    print(f"  Bottom 5 Elo:")
    for tid, r in ranked[-5:]:
        print(f"    {teams.get(tid, tid):24} {r:.0f}")
    con.close()
    print("\nDONE")
