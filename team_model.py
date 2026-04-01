"""
team_model.py
--------------
Trains and runs the team game prediction model.

Predicts:
  - Winner (home/away) + confidence %
  - Home score
  - Away score
  - OT probability

Uses team stats + goalie stats + player aggregates +
HomeIce differential + back-to-back flags.

Ensemble: RandomForest + XGBoost averaged for winner prediction.

Usage:
    from team_model import TeamModel
    model = TeamModel()
    model.train(con)
    pred = model.predict_game(con, home_team_id, away_team_id, season)
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from db_features import (
    get_connection,
    get_completed_games,
    build_game_features,
    get_team_features,
)

MODEL_FILE = "nhl_team_model.pkl"

class TeamModel:

    def __init__(self):
        self.rf_model      = None  # RandomForest winner
        self.gb_model      = None  # GradientBoosting winner
        self.home_score_model = None
        self.away_score_model = None
        self.ot_model      = None
        self.scaler        = None
        self.feature_names = None
        self.trained_date  = None
        self.training_games = 0
        self.cv_accuracy   = 0.0

    # -- Fast feature builder (no DB calls) -------------------
    def _build_game_feats_fast(self, home_id, away_id,
                                teams_df, goalies_df,
                                home_b2b=False, away_b2b=False):
        """Build game feature dict entirely from in-memory DataFrames."""
        home = teams_df[teams_df["TeamID"] == home_id]
        away = teams_df[teams_df["TeamID"] == away_id]
        if home.empty or away.empty:
            return None
        home = home.iloc[0]
        away = away.iloc[0]

        goalies_idx = goalies_df.set_index("TeamID")
        hg = goalies_idx.loc[home_id] if home_id in goalies_idx.index else None
        ag = goalies_idx.loc[away_id] if away_id in goalies_idx.index else None

        def v(row, key, default=0.0):
            if row is None:
                return default
            val = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
            try:
                f = float(val)
                return default if (np.isnan(f) or np.isinf(f)) else f
            except:
                return default

        # HomeIce differential
        homeice = round((v(home, "HomeWinPct", 0.5) - v(away, "AwayWinPct", 0.5)) * 6, 3)

        return {
            "home_CF_Pct":          v(home, "CF_Pct", 50.0),
            "home_FF_Pct":          v(home, "FF_Pct", 50.0),
            "home_xGF_Pct":         v(home, "xGF_Pct", 50.0),
            "home_SCF_Pct":         v(home, "SCF_Pct", 50.0),
            "home_HDCF_Pct":        v(home, "HDCF_Pct", 50.0),
            "home_HDSV_Pct":        v(home, "HDSV_Pct", 85.0),
            "home_PDO":             v(home, "PDO", 100.0),
            "home_SH_Pct":          v(home, "SH_Pct", 9.0),
            "home_SV_Pct":          v(home, "SV_Pct", 0.910),
            "home_GF_Per_Game":     v(home, "GF_Per_Game", 3.0),
            "home_GA_Per_Game":     v(home, "GA_Per_Game", 3.0),
            "home_Points":          v(home, "Points", 50.0),
            "home_Point_Pct":       v(home, "Point_Pct", 0.5),
            "home_HomeWinPct":      v(home, "HomeWinPct", 0.5),
            "home_HomeGF_Per_Game": v(home, "HomeGF_Per_Game", 3.0),
            "home_HomeGA_Per_Game": v(home, "HomeGA_Per_Game", 3.0),
            "home_FaceoffWinPct":   v(home, "Team_FaceoffWinPct", 50.0),
            "home_GV_Per_Game":     v(home, "Team_Giveaways_Per_Game", 10.0),
            "home_TK_Per_Game":     v(home, "Team_Takeaways_Per_Game", 8.0),
            "away_CF_Pct":          v(away, "CF_Pct", 50.0),
            "away_FF_Pct":          v(away, "FF_Pct", 50.0),
            "away_xGF_Pct":         v(away, "xGF_Pct", 50.0),
            "away_SCF_Pct":         v(away, "SCF_Pct", 50.0),
            "away_HDCF_Pct":        v(away, "HDCF_Pct", 50.0),
            "away_HDSV_Pct":        v(away, "HDSV_Pct", 85.0),
            "away_PDO":             v(away, "PDO", 100.0),
            "away_SH_Pct":          v(away, "SH_Pct", 9.0),
            "away_SV_Pct":          v(away, "SV_Pct", 0.910),
            "away_GF_Per_Game":     v(away, "GF_Per_Game", 3.0),
            "away_GA_Per_Game":     v(away, "GA_Per_Game", 3.0),
            "away_Points":          v(away, "Points", 50.0),
            "away_Point_Pct":       v(away, "Point_Pct", 0.5),
            "away_AwayWinPct":      v(away, "AwayWinPct", 0.5),
            "away_AwayGF_Per_Game": v(away, "AwayGF_Per_Game", 3.0),
            "away_AwayGA_Per_Game": v(away, "AwayGA_Per_Game", 3.0),
            "away_FaceoffWinPct":   v(away, "Team_FaceoffWinPct", 50.0),
            "away_GV_Per_Game":     v(away, "Team_Giveaways_Per_Game", 10.0),
            "away_TK_Per_Game":     v(away, "Team_Takeaways_Per_Game", 8.0),
            "home_goalie_GSAX":     v(hg, "Goalie_GSAX", 0.0),
            "home_goalie_GSAX60":   v(hg, "Goalie_GSAX60", 0.0),
            "home_goalie_SV_Pct":   v(hg, "Goalie_SV_Pct", 0.910),
            "home_goalie_HDSV_Pct": v(hg, "Goalie_HDSV_Pct", 85.0),
            "away_goalie_GSAX":     v(ag, "Goalie_GSAX", 0.0),
            "away_goalie_GSAX60":   v(ag, "Goalie_GSAX60", 0.0),
            "away_goalie_SV_Pct":   v(ag, "Goalie_SV_Pct", 0.910),
            "away_goalie_HDSV_Pct": v(ag, "Goalie_HDSV_Pct", 85.0),
            "home_top6_ixG_per60":  0.0,
            "home_top6_iHDCF_per60":0.0,
            "home_top6_SH_pct":     0.0,
            "home_top4D_xGF_pct":   0.0,
            "home_top4D_CF_pct":    0.0,
            "away_top6_ixG_per60":  0.0,
            "away_top6_iHDCF_per60":0.0,
            "away_top6_SH_pct":     0.0,
            "away_top4D_xGF_pct":   0.0,
            "away_top4D_CF_pct":    0.0,
            "home_injured_ixG_lost":0.0,
            "away_injured_ixG_lost":0.0,
            "homeice_differential": homeice,
            "points_differential":  v(home, "Points") - v(away, "Points"),
            "xGF_pct_differential": v(home, "xGF_Pct", 50.0) - v(away, "xGF_Pct", 50.0),
            "GSAX_differential":    v(hg, "Goalie_GSAX", 0.0) - v(ag, "Goalie_GSAX", 0.0),
            "home_is_b2b":          1 if home_b2b else 0,
            "away_is_b2b":          1 if away_b2b else 0,
            "is_playoff":           0,
            # Rolling 10-game — populated in train loop per game
            "home_rolling_win_pct":     0.5,
            "home_rolling_gf_per_game": 3.0,
            "home_rolling_ga_per_game": 3.0,
            "home_rolling_goal_diff":   0.0,
            "home_rolling_shots":       28.0,
            "away_rolling_win_pct":     0.5,
            "away_rolling_gf_per_game": 3.0,
            "away_rolling_ga_per_game": 3.0,
            "away_rolling_goal_diff":   0.0,
            "away_rolling_shots":       28.0,
            "rolling_win_pct_diff":     0.0,
            "rolling_goal_diff_diff":   0.0,
            # Goal differential — populated per game in train loop
            "home_goal_diff":           v(home, "GoalDiff", 0.0),
            "away_goal_diff":           v(away, "GoalDiff", 0.0),
            "goal_diff_differential":   v(home, "GoalDiff", 0.0) - v(away, "GoalDiff", 0.0),
            # Streak — populated per game in train loop
            "home_streak":              0.0,
            "away_streak":              0.0,
            "streak_differential":      0.0,
        }

    # -- Train ------------------------------------------------
    def train(self, con, player_model=None):
        """
        Train on all completed games from both seasons.
        Optionally accepts a trained PlayerModel to include
        player aggregate features.
        """
        print("\n" + "="*60)
        print("TRAINING TEAM MODEL")
        print("="*60)

        games = get_completed_games(con)
        print(f"\n  Completed games available: {len(games)}")

        X_list        = []
        y_winner_list = []
        y_home_list   = []
        y_away_list   = []
        y_ot_list     = []
        skipped       = 0

        # Pre-load data per season to avoid per-game DB queries
        from db_features import get_team_features, get_player_features, get_goalie_features
        season_cache = {}
        for s in ["2024-25", "2025-26"]:
            print(f"  Pre-loading {s} data...")
            season_cache[s] = {
                "teams":   get_team_features(con, s),
                "goalies": get_goalie_features(con, s),
                "players": get_player_features(con, s),
            }
            print(f"    {len(season_cache[s]['teams'])} teams, "
                  f"{len(season_cache[s]['players'])} players loaded")

        # Pre-compute rolling 10-game stats with ONE SQL query
        print("  Pre-computing rolling 10-game stats (batch SQL)...")
        rolling_df = con.execute("""
            WITH game_results AS (
                SELECT
                    g.GameDate,
                    g.HomeTeamID AS TeamID,
                    g.HomeScore  AS GF,
                    g.AwayScore  AS GA,
                    CASE WHEN g.HomeScore > g.AwayScore THEN 1 ELSE 0 END AS Win,
                    COALESCE(gs.Shots, 28) AS Shots
                FROM Games g
                LEFT JOIN GameStats gs ON gs.GameID=g.GameID AND gs.TeamID=g.HomeTeamID
                WHERE g.HomeScore IS NOT NULL
                UNION ALL
                SELECT
                    g.GameDate,
                    g.AwayTeamID AS TeamID,
                    g.AwayScore  AS GF,
                    g.HomeScore  AS GA,
                    CASE WHEN g.AwayScore > g.HomeScore THEN 1 ELSE 0 END AS Win,
                    COALESCE(gs.Shots, 28) AS Shots
                FROM Games g
                LEFT JOIN GameStats gs ON gs.GameID=g.GameID AND gs.TeamID=g.AwayTeamID
                WHERE g.HomeScore IS NOT NULL
            ),
            rolling AS (
                SELECT
                    a.TeamID,
                    a.GameDate AS ForDate,
                    AVG(b.Win)   OVER w AS rolling_win_pct,
                    AVG(b.GF)    OVER w AS rolling_gf,
                    AVG(b.GA)    OVER w AS rolling_ga,
                    AVG(b.GF-b.GA) OVER w AS rolling_goal_diff,
                    AVG(b.Shots) OVER w AS rolling_shots
                FROM game_results a
                JOIN game_results b
                  ON b.TeamID = a.TeamID
                 AND b.GameDate < a.GameDate
                WINDOW w AS (
                    PARTITION BY a.TeamID, a.GameDate
                    ORDER BY b.GameDate DESC
                    ROWS BETWEEN CURRENT ROW AND 9 FOLLOWING
                )
            )
            SELECT DISTINCT
                TeamID,
                ForDate,
                COALESCE(rolling_win_pct, 0.5)  AS rolling_win_pct,
                COALESCE(rolling_gf, 3.0)        AS rolling_gf_per_game,
                COALESCE(rolling_ga, 3.0)        AS rolling_ga_per_game,
                COALESCE(rolling_goal_diff, 0.0) AS rolling_goal_diff,
                COALESCE(rolling_shots, 28.0)    AS rolling_shots
            FROM rolling
        """).df()

        # Build lookup dict: (team_id, date_str) -> rolling stats dict
        from db_features import _default_rolling
        default_rolling = _default_rolling()
        rolling_cache = {}
        for _, row in rolling_df.iterrows():
            key = (int(row["TeamID"]), str(row["ForDate"])[:10])
            rolling_cache[key] = {
                "rolling_win_pct":     float(row["rolling_win_pct"]),
                "rolling_gf_per_game": float(row["rolling_gf_per_game"]),
                "rolling_ga_per_game": float(row["rolling_ga_per_game"]),
                "rolling_goal_diff":   float(row["rolling_goal_diff"]),
                "rolling_shots":       float(row["rolling_shots"]),
            }
        print(f"    {len(rolling_cache)} team/date combinations loaded")

        # Pre-compute streak before each game (consecutive W/L going into that game)
        print("  Pre-computing team streaks (batch)...")
        streak_results = con.execute("""
            SELECT GameDate, HomeTeamID AS TeamID,
                   CASE WHEN HomeScore > AwayScore THEN 1 ELSE -1 END AS result
            FROM Games WHERE HomeScore IS NOT NULL
            UNION ALL
            SELECT GameDate, AwayTeamID,
                   CASE WHEN AwayScore > HomeScore THEN 1 ELSE -1 END
            FROM Games WHERE HomeScore IS NOT NULL
            ORDER BY TeamID, GameDate ASC
        """).df()

        streak_cache = {}
        for team_id, team_df in streak_results.groupby("TeamID"):
            team_df = team_df.sort_values("GameDate")
            streak = 0
            for _, row in team_df.iterrows():
                date_str = str(row["GameDate"])[:10]
                streak_cache[(int(team_id), date_str)] = streak  # streak BEFORE this game
                r = int(row["result"])
                streak = (streak + 1 if streak > 0 else 1) if r == 1 else (streak - 1 if streak < 0 else -1)
        print(f"    {len(streak_cache)} team/date streak entries")

        for _, game in games.iterrows():
            home_id  = int(game["HomeTeamID"])
            away_id  = int(game["AwayTeamID"])
            season   = game["Season"]
            home_b2b = bool(game["HomeIsBackToBack"])
            away_b2b = bool(game["AwayIsBackToBack"])

            cache = season_cache.get(season)
            if cache is None:
                skipped += 1
                continue

            feats = self._build_game_feats_fast(
                home_id, away_id,
                cache["teams"], cache["goalies"],
                home_b2b, away_b2b
            )

            if feats is None:
                skipped += 1
                continue

            # Set playoff flag from game data
            # Use direct Series access with fallback
            try:
                is_playoff_val = int(game["IsPlayoff"]) if "IsPlayoff" in game.index else 0
            except:
                is_playoff_val = 0
            feats["is_playoff"] = is_playoff_val

            # Rolling stats pre-computed in batch above
            game_date = str(game["GameDate"])[:10]
            home_r = rolling_cache.get((home_id, game_date), default_rolling)
            away_r = rolling_cache.get((away_id, game_date), default_rolling)
            feats["home_rolling_win_pct"]     = home_r["rolling_win_pct"]
            feats["home_rolling_gf_per_game"] = home_r["rolling_gf_per_game"]
            feats["home_rolling_ga_per_game"] = home_r["rolling_ga_per_game"]
            feats["home_rolling_goal_diff"]   = home_r["rolling_goal_diff"]
            feats["home_rolling_shots"]       = home_r["rolling_shots"]
            feats["away_rolling_win_pct"]     = away_r["rolling_win_pct"]
            feats["away_rolling_gf_per_game"] = away_r["rolling_gf_per_game"]
            feats["away_rolling_ga_per_game"] = away_r["rolling_ga_per_game"]
            feats["away_rolling_goal_diff"]   = away_r["rolling_goal_diff"]
            feats["away_rolling_shots"]       = away_r["rolling_shots"]
            feats["rolling_win_pct_diff"]     = home_r["rolling_win_pct"] - away_r["rolling_win_pct"]
            feats["rolling_goal_diff_diff"]   = home_r["rolling_goal_diff"] - away_r["rolling_goal_diff"]

            # Streak features
            home_s = float(streak_cache.get((home_id, game_date), 0))
            away_s = float(streak_cache.get((away_id, game_date), 0))
            feats["home_streak"]         = home_s
            feats["away_streak"]         = away_s
            feats["streak_differential"] = home_s - away_s

            # Add player projections using pre-loaded data
            if player_model is not None:
                try:
                    _, home_proj = player_model.predict_team_players(
                        con, home_id, away_id,
                        is_home=True, season=season,
                        players_df=cache["players"],
                        teams_df=cache["teams"],
                        goalies_df=cache["goalies"]
                    )
                    _, away_proj = player_model.predict_team_players(
                        con, away_id, home_id,
                        is_home=False, season=season,
                        players_df=cache["players"],
                        teams_df=cache["teams"],
                        goalies_df=cache["goalies"]
                    )
                    feats["home_projected_goals"] = home_proj
                    feats["away_projected_goals"] = away_proj
                    feats["projected_goals_diff"] = home_proj - away_proj
                except:
                    feats["home_projected_goals"] = 3.0
                    feats["away_projected_goals"] = 3.0
                    feats["projected_goals_diff"] = 0.0

            X_list.append(feats)

            home_score = int(game["HomeScore"])
            away_score = int(game["AwayScore"])
            y_winner_list.append(1 if home_score > away_score else 0)
            y_home_list.append(home_score)
            y_away_list.append(away_score)
            y_ot_list.append(1 if game["OvertimeFlag"] else 0)

        if len(X_list) == 0:
            print("  ERROR: No training data built")
            return

        X = pd.DataFrame(X_list).fillna(0)
        self.feature_names = list(X.columns)

        y_winner = np.array(y_winner_list)
        y_home   = np.array(y_home_list)
        y_away   = np.array(y_away_list)
        y_ot     = np.array(y_ot_list)

        # -- Sample weights — regular season games weighted 2x playoff --
        is_playoff_arr = X["is_playoff"].values if "is_playoff" in X.columns else np.zeros(len(X))
        sample_weights = np.where(is_playoff_arr == 1, 1.0, 2.0)

        reg_count    = int((is_playoff_arr == 0).sum())
        playoff_count= int((is_playoff_arr == 1).sum())

        print(f"\n  Training games:    {len(X)}")
        print(f"    Regular season:  {reg_count}")
        print(f"    Playoff:         {playoff_count} (weighted 0.5x)")
        print(f"  Skipped:           {skipped} (missing team stats)")
        print(f"  Features:          {len(self.feature_names)}")
        print(f"  Home win rate:     {y_winner.mean():.1%}")
        print(f"  OT rate:           {y_ot.mean():.1%}")
        print(f"  Avg home score:    {y_home.mean():.2f}")
        print(f"  Avg away score:    {y_away.mean():.2f}")

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Stratified train/test split — keep home win ratio consistent
        X_tr, X_te, y_w_tr, y_w_te, sw_tr, sw_te = train_test_split(
            X_scaled, y_winner, sample_weights,
            test_size=0.15, random_state=42, stratify=y_winner
        )
        # Align other targets with same split indices
        tr_size = len(X_tr)
        te_size = len(X_te)

        print("\n  Training models...")

        # 1. RandomForest winner — uses sample weights
        print("    [1/5] RandomForest winner...")
        self.rf_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_tr, y_w_tr, sample_weight=sw_tr)
        rf_acc = accuracy_score(y_w_te, self.rf_model.predict(X_te))
        print(f"          RF accuracy: {rf_acc:.1%}")

        # 2. GradientBoosting winner — tuned for this dataset
        # Key fixes: lower max_depth (less overfit), higher min_samples_leaf,
        # more trees with lower LR for better generalization
        print("    [2/5] GradientBoosting winner...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=3,           # shallower = less overfit
            learning_rate=0.02,    # lower LR = more stable
            subsample=0.7,         # more randomness = less overfit
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42
        )
        self.gb_model.fit(X_tr, y_w_tr, sample_weight=sw_tr)
        gb_acc = accuracy_score(y_w_te, self.gb_model.predict(X_te))
        print(f"          GB accuracy: {gb_acc:.1%}")

        # Weighted ensemble — RF gets 60%, GB gets 40%
        # RF is more reliable on smaller datasets
        rf_proba  = self.rf_model.predict_proba(X_te)[:, 1]
        gb_proba  = self.gb_model.predict_proba(X_te)[:, 1]
        ens_proba = (rf_proba * 0.6) + (gb_proba * 0.4)
        ens_pred  = (ens_proba >= 0.5).astype(int)
        ens_acc   = accuracy_score(y_w_te, ens_pred)
        print(f"          Ensemble accuracy (60/40): {ens_acc:.1%}")

        # Cross-validation for more reliable accuracy estimate
        print("          Running 5-fold CV...")
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(
            self.rf_model, X_scaled, y_winner,
            cv=5, scoring="accuracy", n_jobs=-1
        )
        print(f"          CV accuracy: {cv_scores.mean():.1%} "
              f"(±{cv_scores.std():.1%})")
        self.cv_accuracy = ens_acc
        self.cv_mean     = float(cv_scores.mean())
        self.cv_std      = float(cv_scores.std())

        # 3. Home score regressor
        print("    [3/5] Home score regressor...")
        self.home_score_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=8, random_state=42
        )
        self.home_score_model.fit(
            X_scaled[:tr_size], y_home[:tr_size],
            sample_weight=sample_weights[:tr_size]
        )

        # 4. Away score regressor
        print("    [4/5] Away score regressor...")
        self.away_score_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=8, random_state=42
        )
        self.away_score_model.fit(
            X_scaled[:tr_size], y_away[:tr_size],
            sample_weight=sample_weights[:tr_size]
        )

        # 5. OT predictor
        print("    [5/5] Overtime predictor...")
        self.ot_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=8, random_state=42
        )
        self.ot_model.fit(
            X_scaled[:tr_size], y_ot[:tr_size],
            sample_weight=sample_weights[:tr_size]
        )

        # Feature importance
        print("\n  Top 10 most important features (RF):")
        importances = pd.Series(
            self.rf_model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        for feat, imp in importances.head(10).items():
            print(f"    {feat:40} {imp:.4f}")

        self.trained_date  = datetime.now()
        self.training_games = len(X)

        print(f"\n  Team model trained — {self.training_games} games, "
              f"ensemble accuracy: {self.cv_accuracy:.1%}")
        self.save()

    # -- Predict single game -----------------------------------
    def predict_game(self, con, home_team_id, away_team_id,
                     season, home_b2b=False, away_b2b=False,
                     home_proj_goals=None, away_proj_goals=None,
                     is_playoff=False):
        """
        Predicts winner, score, and OT for one game.
        Returns a dict with all prediction fields.
        """
        if self.rf_model is None:
            self.load()

        feats = build_game_features(
            con, home_team_id, away_team_id, season,
            home_b2b, away_b2b
        )
        if feats is None:
            return None

        # Set playoff flag
        feats["is_playoff"] = 1 if is_playoff else 0

        # Add player projections if provided
        if home_proj_goals is not None:
            feats["home_projected_goals"] = home_proj_goals
            feats["away_projected_goals"] = away_proj_goals
            feats["projected_goals_diff"] = home_proj_goals - away_proj_goals
        elif "home_projected_goals" in (self.feature_names or []):
            feats["home_projected_goals"] = 3.0
            feats["away_projected_goals"] = 3.0
            feats["projected_goals_diff"] = 0.0

        X = pd.DataFrame([feats])[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Ensemble winner probability
        rf_prob = self.rf_model.predict_proba(X_scaled)[0]
        gb_prob = self.gb_model.predict_proba(X_scaled)[0]
        avg_prob = (rf_prob + gb_prob) / 2

        home_win_prob = float(avg_prob[1])
        away_win_prob = float(avg_prob[0])

        winner_is_home = home_win_prob >= 0.5
        confidence     = home_win_prob if winner_is_home else away_win_prob

        # Score prediction
        home_score = int(round(float(
            np.clip(self.home_score_model.predict(X_scaled)[0], 1, 9)
        )))
        away_score = int(round(float(
            np.clip(self.away_score_model.predict(X_scaled)[0], 1, 9)
        )))

        # Align score with winner prediction
        if winner_is_home and home_score <= away_score:
            home_score = away_score + 1
        elif not winner_is_home and away_score <= home_score:
            away_score = home_score + 1

        # OT probability
        ot_prob = float(self.ot_model.predict_proba(X_scaled)[0][1])
        is_ot   = ot_prob > 0.45

        # Get team names
        home_name = con.execute(
            f"SELECT TeamName FROM Teams WHERE TeamID = {home_team_id}"
        ).fetchone()[0]
        away_name = con.execute(
            f"SELECT TeamName FROM Teams WHERE TeamID = {away_team_id}"
        ).fetchone()[0]

        return {
            "home_team":        home_name,
            "away_team":        away_name,
            "home_team_id":     home_team_id,
            "away_team_id":     away_team_id,
            "predicted_winner": home_name if winner_is_home else away_name,
            "winner_is_home":   winner_is_home,
            "home_win_prob":    round(home_win_prob, 4),
            "away_win_prob":    round(away_win_prob, 4),
            "confidence":       round(confidence, 4),
            "home_score":       home_score,
            "away_score":       away_score,
            "ot_prob":          round(ot_prob, 4),
            "is_ot":            is_ot,
            "homeice_diff":     round(feats.get("homeice_differential", 0), 3),
            "xGF_diff":         round(feats.get("xGF_pct_differential", 0), 2),
            "GSAX_diff":        round(feats.get("GSAX_differential", 0), 3),
            "home_b2b":         home_b2b,
            "away_b2b":         away_b2b,
        }

    # -- Save / Load -------------------------------------------
    def save(self, path=MODEL_FILE):
        data = {
            "rf_model":          self.rf_model,
            "gb_model":          self.gb_model,
            "home_score_model":  self.home_score_model,
            "away_score_model":  self.away_score_model,
            "ot_model":          self.ot_model,
            "scaler":            self.scaler,
            "feature_names":     self.feature_names,
            "trained_date":      self.trained_date,
            "training_games":    self.training_games,
            "cv_accuracy":       self.cv_accuracy,
            "cv_mean":           getattr(self, "cv_mean", self.cv_accuracy),
            "cv_std":            getattr(self, "cv_std", 0.0),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Team model saved -> {path}")

    def load(self, path=MODEL_FILE):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.rf_model         = data["rf_model"]
        self.gb_model         = data["gb_model"]
        self.home_score_model = data["home_score_model"]
        self.away_score_model = data["away_score_model"]
        self.ot_model         = data["ot_model"]
        self.scaler           = data["scaler"]
        self.feature_names    = data["feature_names"]
        self.trained_date     = data["trained_date"]
        self.training_games   = data["training_games"]
        self.cv_accuracy      = data.get("cv_accuracy", 0.0)
        self.cv_mean          = data.get("cv_mean", self.cv_accuracy)
        self.cv_std           = data.get("cv_std", 0.0)
        print(f"  Team model loaded  (trained {self.trained_date.strftime('%Y-%m-%d')}, "
              f"CV {self.cv_mean:.1%} ±{self.cv_std:.1%})")
        return self


# -- Quick test ------------------------------------------------
if __name__ == "__main__":
    con = get_connection()
    print("Connected!\n")

    model = TeamModel()
    model.train(con)

    print("\nTesting prediction...")
    teams = con.execute("SELECT TeamID, TeamName FROM Teams LIMIT 2").fetchall()
    if len(teams) >= 2:
        pred = model.predict_game(
            con, teams[0][0], teams[1][0], "2025-26"
        )
        if pred:
            print(f"\n  {pred['away_team']} @ {pred['home_team']}")
            print(f"  Winner: {pred['predicted_winner']} ({pred['confidence']:.1%})")
            print(f"  Score:  {pred['away_score']} - {pred['home_score']}")
            print(f"  OT:     {pred['ot_prob']:.1%}")
            print(f"  HomeIce differential: {pred['homeice_diff']}")
            print(f"  xGF differential:     {pred['xGF_diff']}")

    con.close()
    print("\nDone!")
