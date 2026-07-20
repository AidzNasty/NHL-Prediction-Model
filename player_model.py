"""
player_model.py
----------------
Per-game player prop model — POINT-IN-TIME (leak-free) rebuild.

Predicts real per-game probabilities:
  - P(player scores >=1 goal)
  - P(player records >=1 assist)
  - P(player records >=1 point)

History (why this was rewritten, 2026-07):
  The previous model (a) used the player's SEASON average Goals/GP as the
  target while feeding it the same player's season per-60 rates as features
  (predicting a number from itself), and (b) never used per-game data —
  it fabricated rows by sampling random opponents around one season snapshot.
  Its "probabilities" were clipped season rates.

  This version trains on real skater-games from PlayerGameLog. Features are
  built from ONLY each player's earlier games plus the opponent's as-of-date
  goals-against (see pit_player_features.py); targets are the actual per-game
  goal/assist/point events. One calibrated HistGradientBoosting classifier
  per target, so the outputs are genuine, well-calibrated probabilities.

Public interface (train / predict_player / predict_team_players / save / load
and their return keys) is unchanged so train_and_predict.py keeps working.
The teams_df / goalies_df arguments are accepted for compatibility but no
longer used as feature sources (they were season snapshots = leakage).
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss

from db_features import get_connection, get_player_features, get_confirmed_lineup
from pit_player_features import (
    load_player_games, build_opp_defense, prior_season_rates,
    build_player_table, compute_live_player_state, snapshot_player,
    LEAGUE_BASE, _PlayerAcc,
)

MODEL_FILE = "nhl_player_model.pkl"
TARGETS = ["goal", "assist", "point"]


class PlayerModel:

    def __init__(self):
        self.models       = {}   # target -> classifier
        self.calibrators  = {}   # target -> isotonic
        self.feature_names = None
        self.trained_date  = None
        self.training_players = 0
        self.holdout = {}        # target -> (base_ll, model_ll)
        # cached live state
        self._prior = None
        self._accs  = None
        self._opp_ga = None      # (season, team) -> current GA/game
        self._season = None

    @staticmethod
    def _hgb():
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            min_samples_leaf=40, l2_regularization=1.0, random_state=42)

    @staticmethod
    def _oof_proba(model_factory, X_np, y, sw, n_splits=4):
        """Time-series out-of-fold P(y=1) for leak-free calibration."""
        oof = np.full(len(y), np.nan)
        for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X_np):
            m = model_factory().fit(X_np[tr], y[tr], sample_weight=sw[tr])
            oof[te] = m.predict_proba(X_np[te])[:, 1]
        return oof

    # ── Train ─────────────────────────────────────────────────
    def train(self, con):
        print("\n" + "=" * 60)
        print("TRAINING PLAYER MODEL  (point-in-time, per-game)")
        print("=" * 60)

        games   = load_player_games(con)
        opp_def = build_opp_defense(con)
        prior   = prior_season_rates(games)
        X, meta = build_player_table(games, opp_def, prior)
        self.feature_names = list(X.columns)

        dates = pd.to_datetime(meta["GameDate"])
        is_playoff = meta["IsPlayoff"].to_numpy()
        sw = np.where(is_playoff == 1, 0.5, 1.0)
        X_np = X.to_numpy()

        print(f"  Skater-games:  {len(X)}")
        print(f"  Features:      {len(self.feature_names)}")

        # Temporal holdout (last 20% of 2025-26 regular season)
        pool = meta[(meta["Season"] == "2025-26") & (meta["IsPlayoff"] == 0)]
        do_holdout = len(pool) > 500
        if do_holdout:
            cutoff = np.datetime64(pool["GameDate"].quantile(0.80))
            tr_mask = dates.to_numpy() < cutoff
            te_mask = ((dates.to_numpy() >= cutoff)
                       & (meta["Season"] == "2025-26").to_numpy()
                       & (is_playoff == 0))
            print(f"\n  Temporal holdout (n={int(te_mask.sum())}):")

        for tgt in TARGETS:
            y = meta[f"y_{tgt}"].to_numpy()
            if do_holdout:
                m = self._hgb().fit(X_np[tr_mask], y[tr_mask], sample_weight=sw[tr_mask])
                p = m.predict_proba(X_np[te_mask])[:, 1]
                yte = y[te_mask]
                base = yte.mean()
                base_ll = log_loss(yte, np.full(len(yte), base), labels=[0, 1])
                mdl_ll  = log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
                acc     = accuracy_score(yte, (p >= 0.5).astype(int))
                base_acc = max(base, 1 - base)
                self.holdout[tgt] = (float(base_ll), float(mdl_ll))
                print(f"    {tgt:6}  base-rate {base:.3f} | "
                      f"log-loss base->model {base_ll:.4f}->{mdl_ll:.4f} | "
                      f"acc {acc:.3f} (base {base_acc:.3f})")

            # Final model + calibrator on all data
            self.models[tgt] = self._hgb().fit(X_np, y, sample_weight=sw)
            oof = self._oof_proba(self._hgb, X_np, y, sw)
            om = ~np.isnan(oof)
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(oof[om], y[om])
            self.calibrators[tgt] = cal

        self.trained_date = datetime.now()
        self.training_players = len(X)
        self._prior = self._accs = self._opp_ga = self._season = None
        print(f"\n  Player model trained on {self.training_players} skater-games")
        self.save()

    # ── Live-state helper ─────────────────────────────────────
    def _ensure_live_state(self, con):
        if self._accs is not None:
            return
        games = load_player_games(con)
        self._accs  = compute_live_player_state(games)
        self._prior = prior_season_rates(games)
        # current opponent goals-against per game (season-to-date, all completed)
        rows = con.execute("""
            SELECT Season, TeamID, AVG(GA) AS ga_pg FROM (
                SELECT Season, HomeTeamID AS TeamID, AwayScore AS GA
                FROM Games WHERE HomeScore IS NOT NULL
                UNION ALL
                SELECT Season, AwayTeamID, HomeScore
                FROM Games WHERE HomeScore IS NOT NULL
            ) GROUP BY Season, TeamID
        """).df()
        self._opp_ga = {(r.Season, int(r.TeamID)): float(r.ga_pg)
                        for r in rows.itertuples(index=False)}

    def _predict_row(self, feats):
        """Feature dict -> {goal,assist,point}_prob (calibrated)."""
        X = pd.DataFrame([feats])[self.feature_names].to_numpy()
        out = {}
        for tgt in TARGETS:
            raw = float(self.models[tgt].predict_proba(X)[0, 1])
            cal = self.calibrators.get(tgt)
            out[tgt] = float(np.clip(cal.predict([raw])[0], 0.0, 1.0)) if cal else raw
        # A point requires a goal or assist -> keep coherent
        out["point"] = max(out["point"], out["goal"], out["assist"])
        return out

    def _feats_for(self, player_id, opp_team_id, is_home, season, b2b, is_playoff=False):
        acc   = self._accs.get((season, int(player_id)), _PlayerAcc())
        prior = self._prior.get(season, {}).get(int(player_id))
        opp_ga = self._opp_ga.get((season, int(opp_team_id)), 3.0)
        return snapshot_player(acc, prior, opp_ga, is_home, b2b, is_playoff)

    # ── Predict single player ─────────────────────────────────
    def predict_player(self, con, player_id, opp_team_id,
                       is_home, season, b2b=False,
                       players_df=None, teams_df=None, goalies_df=None):
        if not self.models:
            self.load()
        self._ensure_live_state(con)
        feats = self._feats_for(player_id, opp_team_id, is_home, season, b2b)
        p = self._predict_row(feats)
        return {
            "player_id":   int(player_id),
            "goal_prob":   round(p["goal"],   3),
            "assist_prob": round(p["assist"], 3),
            "point_prob":  round(p["point"],  3),
        }

    # ── Predict full team ─────────────────────────────────────
    def predict_team_players(self, con, team_id, opp_team_id,
                             is_home, season, b2b=False, top_n=10,
                             players_df=None, teams_df=None, goalies_df=None):
        """Predict all confirmed-active skaters for a team. Roster/names come
        from players_df; probabilities come from point-in-time game-log state."""
        if not self.models:
            self.load()
        self._ensure_live_state(con)

        if players_df is None:
            players_df = get_player_features(con, season)

        lineup = get_confirmed_lineup(con, team_id)
        team_players = players_df[players_df["TeamID"] == team_id]

        results = []
        for _, player in team_players.iterrows():
            pid = int(player["PlayerID"])
            status = lineup.get(pid, {}).get("status", "Active")
            if status in ("Injured", "Out", "Healthy Scratch"):
                continue

            feats = self._feats_for(pid, opp_team_id, is_home, season, b2b)
            p = self._predict_row(feats)

            results.append({
                "player_id":    pid,
                "player_name":  player["PlayerName"],
                "position":     player["Position"],
                "toi_per_game": player.get("TOI_Per_Game", 0.0),
                "status":       status,
                "goal_prob":    round(p["goal"],   3),
                "assist_prob":  round(p["assist"], 3),
                "point_prob":   round(p["point"],  3),
                "ixG_per60":    float(player.get("ixG_Per60", 0.0) or 0.0),
            })

        if not results:
            return [], 0.0

        df = pd.DataFrame(results).sort_values("point_prob", ascending=False)
        forwards = df[df["position"].isin(["C", "L", "LW", "R", "RW", "F"])]
        # Sum of P(>=1 goal) over the top forwards ~ expected number of goal
        # scorers, a reasonable projected-goals proxy.
        team_proj_goals = float(forwards.head(12)["goal_prob"].sum())
        return df.head(top_n).to_dict("records"), round(team_proj_goals, 2)

    # ── Save / Load ───────────────────────────────────────────
    def save(self, path=MODEL_FILE):
        with open(path, "wb") as f:
            pickle.dump({
                "models":        self.models,
                "calibrators":   self.calibrators,
                "feature_names": self.feature_names,
                "trained_date":  self.trained_date,
                "training_players": self.training_players,
                "holdout":       self.holdout,
            }, f)
        print(f"  Player model saved -> {path}")

    def load(self, path=MODEL_FILE):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.models        = d["models"]
        self.calibrators   = d.get("calibrators", {})
        self.feature_names = d["feature_names"]
        self.trained_date  = d["trained_date"]
        self.training_players = d["training_players"]
        self.holdout       = d.get("holdout", {})
        self._prior = self._accs = self._opp_ga = self._season = None
        td = self.trained_date.strftime("%Y-%m-%d") if self.trained_date else "?"
        print(f"  Player model loaded (trained {td})")
        return self


if __name__ == "__main__":
    con = get_connection()
    model = PlayerModel()
    model.train(con)
    con.close()
    print("Done!")
