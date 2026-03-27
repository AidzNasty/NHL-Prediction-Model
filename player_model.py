"""
player_model.py
----------------
Trains and runs the player performance prediction model.

Predicts per-player per-game probabilities:
  - Goal probability
  - Assist probability
  - Point probability

Key optimization: ALL data loaded upfront in batch.
Zero per-player DB queries during training.
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from db_features import (
    get_connection,
    get_player_features,
    get_team_features,
    get_goalie_features,
    get_confirmed_lineup,
)

MODEL_FILE = "nhl_player_model.pkl"

class PlayerModel:

    def __init__(self):
        self.goal_model    = None
        self.assist_model  = None
        self.point_model   = None
        self.scaler        = None
        self.feature_names = None
        self.trained_date  = None
        self.training_players = 0

    # -- Fast feature builder (no DB calls) -------------------
    def _build_feats(self, player, opp, opp_goalie, is_home):
        """Build feature dict entirely from in-memory DataFrames."""
        def v(row, key, default=0.0):
            if row is None:
                return default
            val = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
            try:
                f = float(val)
                return default if (np.isnan(f) or np.isinf(f)) else f
            except:
                return default

        return {
            # Player individual
            "ixG_Per60":       v(player, "ixG_Per60"),
            "iHDCF_Per60":     v(player, "iHDCF_Per60"),
            "iCF_Per60":       v(player, "iCF_Per60"),
            "iSCF_Per60":      v(player, "iSCF_Per60"),
            "Rush_Per60":      v(player, "Rush_Per60"),
            "Goals_Per60":     v(player, "Goals_Per60"),
            "Assists_Per60":   v(player, "Assists_Per60"),
            "Points_Per60":    v(player, "Points_Per60"),
            "Player_SH_Pct":   v(player, "Player_SH_Pct", 9.0),
            "IPP":             v(player, "IPP", 60.0),
            "TOI_Per_Game":    v(player, "TOI_Per_Game", 12.0),
            "Off_Zone_Start_Pct": v(player, "Off_Zone_Start_Pct", 50.0),
            "OnIce_xGF_Pct":   v(player, "OnIce_xGF_Pct", 50.0),
            "OnIce_CF_Pct":    v(player, "OnIce_CF_Pct", 50.0),
            "OnIce_PDO":       v(player, "OnIce_PDO", 100.0),
            # Opponent defense
            "opp_xGA":         v(opp, "xGA", 150.0),
            "opp_HDCF_Pct":    v(opp, "HDCF_Pct", 50.0),
            "opp_SV_Pct":      v(opp, "SV_Pct", 0.910),
            "opp_GA_Per_Game": v(opp, "GA_Per_Game", 3.0),
            # Opponent goalie
            "opp_goalie_GSAX":     v(opp_goalie, "Goalie_GSAX", 0.0),
            "opp_goalie_HDSV_Pct": v(opp_goalie, "Goalie_HDSV_Pct", 85.0),
            "opp_goalie_SV_Pct":   v(opp_goalie, "Goalie_SV_Pct", 0.910),
            # Context
            "is_home": 1 if is_home else 0,
            "is_b2b":  0,
        }

    # -- Train ------------------------------------------------
    def train(self, con):
        print("\n" + "="*60)
        print("TRAINING PLAYER MODEL")
        print("="*60)

        X_list, y_goal_list, y_assist_list, y_point_list = [], [], [], []

        for season in ["2024-25", "2025-26"]:
            print(f"\n  Loading {season} data (batch)...")

            # ONE query per table per season — no loops hitting DB
            players_df = get_player_features(con, season)
            teams_df   = get_team_features(con, season)
            goalies_df = get_goalie_features(con, season)

            if players_df.empty:
                print(f"  No data for {season}, skipping")
                continue

            print(f"  {len(players_df)} players, {len(teams_df)} teams loaded")

            # Index for fast lookup
            goalies_idx = goalies_df.set_index("TeamID")

            for _, player in players_df.iterrows():
                if player["GP"] < 10:
                    continue

                team_id = int(player["TeamID"])
                gp      = player["GP"]

                # Sample 3 opponents — all in memory
                opps = teams_df[teams_df["TeamID"] != team_id].sample(
                    min(3, len(teams_df) - 1), random_state=42
                )

                for _, opp in opps.iterrows():
                    opp_id     = int(opp["TeamID"])
                    opp_goalie = goalies_idx.loc[opp_id] if opp_id in goalies_idx.index else None

                    for is_home in [True, False]:
                        feats = self._build_feats(player, opp, opp_goalie, is_home)
                        X_list.append(feats)
                        y_goal_list.append(  min(player["Goals"]         / gp, 1.0))
                        y_assist_list.append(min(player["Total_Assists"] / gp, 1.0))
                        y_point_list.append( min(player["Total_Points"]  / gp, 1.0))

        if not X_list:
            print("  ERROR: No training data built")
            return

        X        = pd.DataFrame(X_list).fillna(0)
        self.feature_names = list(X.columns)
        y_goal   = np.array(y_goal_list)
        y_assist = np.array(y_assist_list)
        y_point  = np.array(y_point_list)

        print(f"\n  Training rows:   {len(X)}")
        print(f"  Features:        {len(self.feature_names)}")
        print(f"  Avg goal rate:   {y_goal.mean():.3f}")
        print(f"  Avg assist rate: {y_assist.mean():.3f}")
        print(f"  Avg point rate:  {y_point.mean():.3f}")

        self.scaler  = StandardScaler()
        X_scaled     = self.scaler.fit_transform(X)

        params = dict(n_estimators=100, max_depth=4,
                      learning_rate=0.05, subsample=0.8, random_state=42)

        print("\n  Training goal model...")
        self.goal_model = GradientBoostingRegressor(**params)
        self.goal_model.fit(X_scaled, y_goal)

        print("  Training assist model...")
        self.assist_model = GradientBoostingRegressor(**params)
        self.assist_model.fit(X_scaled, y_assist)

        print("  Training point model...")
        self.point_model = GradientBoostingRegressor(**params)
        self.point_model.fit(X_scaled, y_point)

        self.trained_date     = datetime.now()
        self.training_players = len(X)

        print(f"\n  Player model trained on {self.training_players} samples")
        self.save()

    # -- Predict single player ---------------------------------
    def predict_player(self, con, player_id, opp_team_id,
                       is_home, season, b2b=False,
                       players_df=None, teams_df=None, goalies_df=None):
        """Predict for one player. Accepts pre-loaded DataFrames to avoid DB hits."""
        if self.goal_model is None:
            self.load()

        if players_df is None:
            players_df = get_player_features(con, season)
        if teams_df is None:
            teams_df = get_team_features(con, season)
        if goalies_df is None:
            goalies_df = get_goalie_features(con, season)

        player = players_df[players_df["PlayerID"] == player_id]
        if player.empty:
            return None
        player = player.iloc[0]

        opp = teams_df[teams_df["TeamID"] == opp_team_id]
        if opp.empty:
            return None
        opp = opp.iloc[0]

        goalies_idx = goalies_df.set_index("TeamID")
        opp_goalie  = goalies_idx.loc[opp_team_id] if opp_team_id in goalies_idx.index else None

        feats    = self._build_feats(player, opp, opp_goalie, is_home)
        feats["is_b2b"] = 1 if b2b else 0

        X        = pd.DataFrame([feats])[self.feature_names]
        X_scaled = self.scaler.transform(X)

        goal_prob   = float(np.clip(self.goal_model.predict(X_scaled)[0],   0, 1))
        assist_prob = float(np.clip(self.assist_model.predict(X_scaled)[0], 0, 1))
        point_prob  = float(np.clip(self.point_model.predict(X_scaled)[0],  0, 1))
        point_prob  = max(point_prob, goal_prob, assist_prob)

        return {
            "player_id":   player_id,
            "goal_prob":   round(goal_prob,   3),
            "assist_prob": round(assist_prob, 3),
            "point_prob":  round(point_prob,  3),
        }

    # -- Predict full team -------------------------------------
    def predict_team_players(self, con, team_id, opp_team_id,
                              is_home, season, b2b=False, top_n=10,
                              players_df=None, teams_df=None, goalies_df=None):
        """Predict all confirmed active players for a team. Batch-friendly."""
        if self.goal_model is None:
            self.load()

        # Load data once if not passed in
        if players_df is None:
            players_df = get_player_features(con, season)
        if teams_df is None:
            teams_df = get_team_features(con, season)
        if goalies_df is None:
            goalies_df = get_goalie_features(con, season)

        lineup      = get_confirmed_lineup(con, team_id)
        team_players = players_df[players_df["TeamID"] == team_id]

        opp = teams_df[teams_df["TeamID"] == opp_team_id]
        opp_row = opp.iloc[0] if not opp.empty else None

        goalies_idx = goalies_df.set_index("TeamID")
        opp_goalie  = goalies_idx.loc[opp_team_id] if opp_team_id in goalies_idx.index else None

        results = []
        for _, player in team_players.iterrows():
            pid    = int(player["PlayerID"])
            status = lineup.get(pid, {}).get("status", "Active")
            if status in ("Injured", "Out", "Healthy Scratch"):
                continue

            feats = self._build_feats(player, opp_row, opp_goalie, is_home)
            feats["is_b2b"] = 1 if b2b else 0

            X        = pd.DataFrame([feats])[self.feature_names].fillna(0)
            X_scaled = self.scaler.transform(X)

            goal_prob   = float(np.clip(self.goal_model.predict(X_scaled)[0],   0, 1))
            assist_prob = float(np.clip(self.assist_model.predict(X_scaled)[0], 0, 1))
            point_prob  = float(np.clip(self.point_model.predict(X_scaled)[0],  0, 1))
            point_prob  = max(point_prob, goal_prob, assist_prob)

            results.append({
                "player_id":   pid,
                "player_name": player["PlayerName"],
                "position":    player["Position"],
                "toi_per_game":player["TOI_Per_Game"],
                "status":      status,
                "goal_prob":   round(goal_prob,   3),
                "assist_prob": round(assist_prob, 3),
                "point_prob":  round(point_prob,  3),
                "ixG_per60":   player["ixG_Per60"],
            })

        if not results:
            return [], 0.0

        df = pd.DataFrame(results).sort_values("point_prob", ascending=False)
        forwards = df[df["position"].isin(["C","L","LW","R","RW","F"])]
        team_proj_goals = float(forwards.head(12)["goal_prob"].sum())

        return df.head(top_n).to_dict("records"), round(team_proj_goals, 2)

    # -- Save / Load -------------------------------------------
    def save(self, path=MODEL_FILE):
        with open(path, "wb") as f:
            pickle.dump({
                "goal_model":       self.goal_model,
                "assist_model":     self.assist_model,
                "point_model":      self.point_model,
                "scaler":           self.scaler,
                "feature_names":    self.feature_names,
                "trained_date":     self.trained_date,
                "training_players": self.training_players,
            }, f)
        print(f"  Player model saved -> {path}")

    def load(self, path=MODEL_FILE):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.goal_model       = d["goal_model"]
        self.assist_model     = d["assist_model"]
        self.point_model      = d["point_model"]
        self.scaler           = d["scaler"]
        self.feature_names    = d["feature_names"]
        self.trained_date     = d["trained_date"]
        self.training_players = d["training_players"]
        print(f"  Player model loaded (trained {self.trained_date.strftime('%Y-%m-%d')})")
        return self


if __name__ == "__main__":
    con = get_connection()
    model = PlayerModel()
    model.train(con)
    con.close()
    print("Done!")
