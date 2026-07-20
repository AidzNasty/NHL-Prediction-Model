"""
team_model.py
--------------
Team game prediction model — POINT-IN-TIME (leak-free) rebuild.

Predicts:
  - Winner (home/away) + confidence %
  - Home score / Away score
  - OT probability

History (why this was rewritten, 2026-07):
  The previous version described every historical game with a single
  end-of-season TeamStandings snapshot (CF%, xGF%, PDO, goalie GSAX ...).
  Those aggregates are season-FINAL, so training leaked future information
  and the model used a RANDOM train/test split — reported ~55-58% in CV but
  only 49% live (below the 52% always-home baseline).

  This version builds every feature from `Games` history strictly BEFORE
  each game's date (see pit_features.py): Elo, season-to-date record/scoring,
  home/road splits, rolling-10 form, rest days. Reported accuracy now comes
  from a TEMPORAL holdout, so it reflects real forecasting skill.
  Walk-forward validation (backtest.py) put the winner ensemble at ~54.7%
  on 2025-26, above both the always-home (52.2%) and Elo-only (54.1%)
  baselines.

Public interface is unchanged (train / predict_game / save / load and the
predict_game return keys) so train_and_predict.py and the dashboard keep
working. NOTE: advanced-stat diagnostics (xGF_diff, GSAX_diff) no longer
exist as source data; they are now populated with point-in-time analogs
(goal-diff-per-game edge and Elo edge) — the dashboard tiles can be
relabeled in a later pass.

Usage:
    from team_model import TeamModel
    m = TeamModel(); m.train(con)
    pred = m.predict_game(con, home_id, away_id, "2025-26")
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

from pit_features import (
    load_games, build_feature_table, compute_live_elo,
    build_live_feature_row,
)

MODEL_FILE = "nhl_team_model.pkl"

# Ensemble blend weights for the winner probability (validated in backtest.py)
W_RF, W_GB, W_LR = 0.4, 0.4, 0.2


class TeamModel:

    def __init__(self):
        self.rf_model         = None
        self.gb_model         = None
        self.lr_model         = None
        self.home_score_model = None
        self.away_score_model = None
        self.ot_model         = None
        self.scaler           = None
        self.calibrator       = None   # isotonic map: raw ensemble prob -> calibrated
        self.feature_names    = None
        self.trained_date     = None
        self.training_games   = 0
        # Reported metrics (temporal holdout)
        self.cv_accuracy      = 0.0   # kept name for compat = holdout accuracy
        self.cv_mean          = 0.0
        self.cv_std           = 0.0
        self.holdout_baseline_home = 0.0
        self.holdout_baseline_elo  = 0.0
        # Live prediction state (built lazily, cached per process/run)
        self._live_elo    = None
        self._live_accs   = None
        self._live_season = None

    # ── model factories ───────────────────────────────────────
    @staticmethod
    def _rf():
        return RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=15,
            max_features="sqrt", random_state=42, n_jobs=-1)

    @staticmethod
    def _gb():
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, min_samples_leaf=20, random_state=42)

    @staticmethod
    def _lr():
        return LogisticRegression(C=0.3, max_iter=1000)

    @staticmethod
    def _pois():
        # Expected-goals regressor with Poisson loss (counts are non-negative).
        return HistGradientBoostingRegressor(
            loss="poisson", max_depth=3, learning_rate=0.05,
            max_iter=300, min_samples_leaf=20, l2_regularization=1.0,
            random_state=42)

    def _ensemble_proba(self, X_scaled):
        """Blended RAW P(home win) from the three winner models."""
        p = (W_RF * self.rf_model.predict_proba(X_scaled)[:, 1]
             + W_GB * self.gb_model.predict_proba(X_scaled)[:, 1]
             + W_LR * self.lr_model.predict_proba(X_scaled)[:, 1])
        return p

    def _calibrated_proba(self, X_scaled):
        """Calibrated P(home win). Falls back to raw if no calibrator."""
        raw = self._ensemble_proba(X_scaled)
        if self.calibrator is None:
            return raw
        cal = self.calibrator.predict(raw)
        return np.clip(cal, 0.02, 0.98)

    # ── Out-of-fold ensemble probs (time-series, leak-free) ───
    def _oof_ensemble(self, X_np, y, sw, n_splits=5):
        """
        Time-ordered out-of-fold ensemble probabilities. Each fold trains
        base models only on earlier games and predicts the next block, so
        the probabilities used to fit the calibrator are genuinely
        out-of-sample. Returns oof array (NaN where never predicted).
        """
        oof = np.full(len(y), np.nan)
        tss = TimeSeriesSplit(n_splits=n_splits)
        for tr, te in tss.split(X_np):
            sc = StandardScaler().fit(X_np[tr])
            Xtr, Xte = sc.transform(X_np[tr]), sc.transform(X_np[te])
            rf = self._rf().fit(Xtr, y[tr], sample_weight=sw[tr])
            gb = self._gb().fit(Xtr, y[tr], sample_weight=sw[tr])
            lr = self._lr().fit(Xtr, y[tr], sample_weight=sw[tr])
            oof[te] = (W_RF * rf.predict_proba(Xte)[:, 1]
                       + W_GB * gb.predict_proba(Xte)[:, 1]
                       + W_LR * lr.predict_proba(Xte)[:, 1])
        return oof

    @staticmethod
    def _fit_calibrator(raw_probs, outcomes):
        """Isotonic calibrator mapping raw ensemble prob -> P(home win)."""
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(raw_probs, outcomes)
        return cal

    @staticmethod
    def _poisson_scoreline(lam_h, lam_a, winner_is_home, is_ot, maxg=10):
        """
        Most-likely scoreline from two independent-Poisson goal counts,
        conditioned on the predicted winner. If OT is predicted, restrict
        to a one-goal margin (OT/SO games are decided by exactly one goal).
        Returns (home_score, away_score) as ints.
        """
        gs = np.arange(maxg + 1)
        ph = poisson.pmf(gs, max(lam_h, 0.05))
        pa = poisson.pmf(gs, max(lam_a, 0.05))
        grid = np.outer(ph, pa)                 # grid[i, j] = P(home=i, away=j)
        ii, jj = np.indices(grid.shape)

        if winner_is_home:
            mask = ii > jj
        else:
            mask = jj > ii
        if is_ot:
            mask = mask & (np.abs(ii - jj) == 1)

        m = grid * mask
        if m.sum() <= 0:
            # Fallback: round expected goals, then enforce winner (and OT margin)
            i, j = int(round(lam_h)), int(round(lam_a))
            if winner_is_home and i <= j:
                i = j + 1
            elif not winner_is_home and j <= i:
                j = i + 1
            return max(0, i), max(0, j)
        idx = np.unravel_index(np.argmax(m), m.shape)
        return int(idx[0]), int(idx[1])

    # ── Train ─────────────────────────────────────────────────
    def train(self, con, player_model=None):
        """
        Build leak-free point-in-time features from Games, report a TEMPORAL
        holdout metric, then fit final models on all history.

        `player_model` is accepted for interface compatibility but is
        intentionally NOT used as a feature source — player aggregates are
        season snapshots and would reintroduce leakage. Player projections
        are still written separately by train_and_predict.py.
        """
        print("\n" + "=" * 60)
        print("TRAINING TEAM MODEL  (point-in-time features)")
        print("=" * 60)

        games = load_games(con)
        X, meta = build_feature_table(games)
        self.feature_names = list(X.columns)

        y_winner = meta["y_home_win"].values
        y_home   = meta["HomeScore"].values
        y_away   = meta["AwayScore"].values
        y_ot     = meta["OvertimeFlag"].values
        dates    = pd.to_datetime(meta["GameDate"])
        is_playoff = meta["IsPlayoff"].values

        # Down-weight playoffs (different game; smaller sample)
        sample_weights = np.where(is_playoff == 1, 0.5, 1.0)

        print(f"  Games:            {len(X)}")
        print(f"  Features:         {len(self.feature_names)}")
        print(f"  Home win rate:    {y_winner.mean():.1%}")
        print(f"  OT rate:          {y_ot.mean():.1%}")

        # ── Temporal holdout for HONEST reporting ─────────────
        # Test = last ~20% of 2025-26 regular-season games (by date).
        test_pool = meta[(meta["Season"] == "2025-26") & (meta["IsPlayoff"] == 0)]
        if len(test_pool) > 50:
            cutoff = test_pool["GameDate"].quantile(0.80)
            is_2526 = (meta["Season"] == "2025-26").to_numpy()
            dates_np = dates.to_numpy()
            cutoff_np = np.datetime64(cutoff)
            train_mask = dates_np < cutoff_np
            test_mask  = (dates_np >= cutoff_np) & is_2526 & (is_playoff == 0)
            self._report_holdout(X, y_winner, sample_weights,
                                 train_mask, test_mask, cutoff)
            self._report_score_holdout(X, y_home, y_away, sample_weights,
                                       train_mask, test_mask)
        else:
            print("  (Not enough 2025-26 games for a temporal holdout — "
                  "skipping report)")

        # ── Fit FINAL models on ALL history ───────────────────
        print("\n  Fitting final models on all history...")
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.rf_model = self._rf().fit(Xs, y_winner, sample_weight=sample_weights)
        self.gb_model = self._gb().fit(Xs, y_winner, sample_weight=sample_weights)
        self.lr_model = self._lr().fit(Xs, y_winner, sample_weight=sample_weights)

        # Probability calibration — fit isotonic map on time-series OOF probs
        # (base models retrained per fold on earlier games only => leak-free).
        print("  Calibrating probabilities (time-series OOF)...")
        X_np = X.to_numpy()
        oof = self._oof_ensemble(X_np, y_winner, sample_weights)
        m = ~np.isnan(oof)
        self.calibrator = self._fit_calibrator(oof[m], y_winner[m])
        raw_ll = log_loss(y_winner[m], np.clip(oof[m], 1e-6, 1 - 1e-6))
        cal_ll = log_loss(y_winner[m],
                          np.clip(self.calibrator.predict(oof[m]), 1e-6, 1 - 1e-6))
        print(f"    OOF log-loss raw -> calibrated: {raw_ll:.4f} -> {cal_ll:.4f}")

        # Poisson expected-goals models (predict lambda_home / lambda_away)
        self.home_score_model = self._pois().fit(Xs, y_home, sample_weight=sample_weights)
        self.away_score_model = self._pois().fit(Xs, y_away, sample_weight=sample_weights)

        self.ot_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=15, random_state=42
        ).fit(Xs, y_ot, sample_weight=sample_weights)

        # Feature importance (from RF)
        print("\n  Top 12 features (RF importance):")
        imp = pd.Series(self.rf_model.feature_importances_,
                        index=self.feature_names).sort_values(ascending=False)
        for feat, val in imp.head(12).items():
            print(f"    {feat:24} {val:.4f}")

        self.trained_date   = datetime.now()
        self.training_games = len(X)
        # Refresh cached live state after retraining
        self._live_elo = self._live_accs = self._live_season = None

        print(f"\n  Team model trained — {self.training_games} games, "
              f"holdout accuracy: {self.cv_mean:.1%}")
        self.save()

    def _report_holdout(self, X, y, sw, train_mask, test_mask, cutoff):
        """Fit on pre-cutoff games, evaluate on the held-out tail."""
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_mask])
        ytr, yte = y[train_mask], y[test_mask]
        swtr = sw[train_mask]

        rf = self._rf().fit(Xtr, ytr, sample_weight=swtr)
        gb = self._gb().fit(Xtr, ytr, sample_weight=swtr)
        lr = self._lr().fit(Xtr, ytr, sample_weight=swtr)
        p = (W_RF * rf.predict_proba(Xte)[:, 1]
             + W_GB * gb.predict_proba(Xte)[:, 1]
             + W_LR * lr.predict_proba(Xte)[:, 1])
        pred = (p >= 0.5).astype(int)

        # Fit a calibrator on OOF probs from the PRE-cutoff data only, then
        # apply it to the holdout so the calibrated metrics are honest.
        Xtr_np = X.to_numpy()[train_mask]
        oof = self._oof_ensemble(Xtr_np, ytr, swtr)
        om = ~np.isnan(oof)
        cal = self._fit_calibrator(oof[om], ytr[om])
        p_cal = np.clip(cal.predict(p), 0.02, 0.98)

        acc        = accuracy_score(yte, pred)
        home_base  = yte.mean()
        elo_p      = X["elo_home_prob"].to_numpy()[test_mask]
        elo_acc    = accuracy_score(yte, (elo_p >= 0.5).astype(int))
        ll         = log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))
        ll_cal     = log_loss(yte, p_cal)
        brier      = brier_score_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))
        brier_cal  = brier_score_loss(yte, p_cal)

        self.cv_accuracy = acc
        self.cv_mean     = acc
        self.cv_std      = 0.0
        self.holdout_baseline_home = home_base
        self.holdout_baseline_elo  = elo_acc

        print(f"\n  Temporal holdout (games on/after {pd.Timestamp(cutoff).date()}, "
              f"n={int(test_mask.sum())}):")
        print(f"    Always-home baseline : {home_base:.1%}")
        print(f"    Elo-only baseline    : {elo_acc:.1%}")
        print(f"    >> MODEL ensemble    : {acc:.1%}  "
              f"({acc - home_base:+.1%} vs home, {acc - elo_acc:+.1%} vs Elo)")
        print(f"    Log-loss  raw -> cal : {ll:.4f} -> {ll_cal:.4f}")
        print(f"    Brier     raw -> cal : {brier:.4f} -> {brier_cal:.4f}")

    def _report_score_holdout(self, X, y_home, y_away, sw, train_mask, test_mask):
        """Honest score MAE: fit Poisson goal models on pre-cutoff games only,
        evaluate expected-goals MAE on the holdout vs a mean-goals baseline."""
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_mask])
        hm = self._pois().fit(Xtr, y_home[train_mask], sample_weight=sw[train_mask])
        am = self._pois().fit(Xtr, y_away[train_mask], sample_weight=sw[train_mask])
        lam_h = hm.predict(Xte)
        lam_a = am.predict(Xte)
        mae_h = mean_absolute_error(y_home[test_mask], lam_h)
        mae_a = mean_absolute_error(y_away[test_mask], lam_a)
        base_h = mean_absolute_error(y_home[test_mask],
                                     np.full(test_mask.sum(), y_home[train_mask].mean()))
        base_a = mean_absolute_error(y_away[test_mask],
                                     np.full(test_mask.sum(), y_away[train_mask].mean()))
        print(f"    Score MAE (home/away): {mae_h:.2f}/{mae_a:.2f} goals  "
              f"(mean baseline {base_h:.2f}/{base_a:.2f})")

    # ── Live-state helper ─────────────────────────────────────
    def _ensure_live_state(self, con):
        """Build (and cache) current Elo + accumulators for predicting
        upcoming games. Safe to cache within a run: Games history does not
        change while today's predictions are being written."""
        if self._live_elo is None:
            games = load_games(con)
            self._live_elo, self._live_accs, self._live_season = \
                compute_live_elo(games)

    # ── Predict single game ───────────────────────────────────
    def predict_game(self, con, home_team_id, away_team_id,
                     season, home_b2b=False, away_b2b=False,
                     home_proj_goals=None, away_proj_goals=None,
                     is_playoff=False, game_date=None):
        """
        Predicts winner, score, and OT for one upcoming game.
        Returns a dict (same keys as before). `home_proj_goals` /
        `away_proj_goals` are accepted for interface compat but unused.
        """
        if self.rf_model is None:
            self.load()
        self._ensure_live_state(con)

        if game_date is None:
            game_date = datetime.now().date()

        feats = build_live_feature_row(
            self._live_elo, self._live_accs, season,
            int(home_team_id), int(away_team_id), game_date,
            home_b2b=home_b2b, away_b2b=away_b2b, is_playoff=is_playoff,
        )

        X = pd.DataFrame([feats])[self.feature_names].fillna(0)
        Xs = self.scaler.transform(X)

        home_win_prob = float(self._calibrated_proba(Xs)[0])
        away_win_prob = 1.0 - home_win_prob
        winner_is_home = home_win_prob >= 0.5
        confidence     = home_win_prob if winner_is_home else away_win_prob

        ot_prob = float(self.ot_model.predict_proba(Xs)[0][1])
        is_ot   = ot_prob > 0.45

        # Poisson expected goals -> coherent joint scoreline (winner-consistent,
        # OT-aware). Replaces the old +1 alignment hack.
        lam_h = float(np.clip(self.home_score_model.predict(Xs)[0], 0.5, 9))
        lam_a = float(np.clip(self.away_score_model.predict(Xs)[0], 0.5, 9))
        home_score, away_score = self._poisson_scoreline(
            lam_h, lam_a, winner_is_home, is_ot)

        home_name = con.execute(
            f"SELECT TeamName FROM Teams WHERE TeamID = {int(home_team_id)}"
        ).fetchone()[0]
        away_name = con.execute(
            f"SELECT TeamName FROM Teams WHERE TeamID = {int(away_team_id)}"
        ).fetchone()[0]

        # Point-in-time diagnostics (replace old advanced-stat diffs):
        #   homeice_diff : same formula as the original Excel model, now as-of-date
        #   xGF_diff     : season-to-date goal-differential-per-game edge (scoring proxy)
        #   GSAX_diff    : Elo rating edge (team-strength proxy), scaled to ~single digits
        homeice_diff = round((feats["home_home_win_pct"] - feats["away_away_win_pct"]) * 6, 3)
        xgf_diff     = round(feats["goal_diff_pg_diff"], 2)
        gsax_diff    = round(feats["elo_diff"] / 100.0, 2)

        return {
            "home_team":        home_name,
            "away_team":        away_name,
            "home_team_id":     int(home_team_id),
            "away_team_id":     int(away_team_id),
            "predicted_winner": home_name if winner_is_home else away_name,
            "winner_is_home":   winner_is_home,
            "home_win_prob":    round(home_win_prob, 4),
            "away_win_prob":    round(away_win_prob, 4),
            "confidence":       round(confidence, 4),
            "home_score":       home_score,
            "away_score":       away_score,
            "ot_prob":          round(ot_prob, 4),
            "is_ot":            is_ot,
            "homeice_diff":     homeice_diff,
            "xGF_diff":         xgf_diff,
            "GSAX_diff":        gsax_diff,
            "elo_diff":         round(feats["elo_diff"], 1),
            "elo_home_prob":    round(feats["elo_home_prob"], 4),
            "home_b2b":         home_b2b,
            "away_b2b":         away_b2b,
        }

    # ── Save / Load ───────────────────────────────────────────
    def save(self, path=MODEL_FILE):
        data = {
            "rf_model":          self.rf_model,
            "gb_model":          self.gb_model,
            "lr_model":          self.lr_model,
            "home_score_model":  self.home_score_model,
            "away_score_model":  self.away_score_model,
            "ot_model":          self.ot_model,
            "scaler":            self.scaler,
            "calibrator":        self.calibrator,
            "feature_names":     self.feature_names,
            "trained_date":      self.trained_date,
            "training_games":    self.training_games,
            "cv_accuracy":       self.cv_accuracy,
            "cv_mean":           self.cv_mean,
            "cv_std":            self.cv_std,
            "holdout_baseline_home": self.holdout_baseline_home,
            "holdout_baseline_elo":  self.holdout_baseline_elo,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Team model saved -> {path}")

    def load(self, path=MODEL_FILE):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.rf_model         = data["rf_model"]
        self.gb_model         = data["gb_model"]
        self.lr_model         = data.get("lr_model")
        self.home_score_model = data["home_score_model"]
        self.away_score_model = data["away_score_model"]
        self.ot_model         = data["ot_model"]
        self.scaler           = data["scaler"]
        self.calibrator       = data.get("calibrator")
        self.feature_names    = data["feature_names"]
        self.trained_date     = data["trained_date"]
        self.training_games   = data["training_games"]
        self.cv_accuracy      = data.get("cv_accuracy", 0.0)
        self.cv_mean          = data.get("cv_mean", self.cv_accuracy)
        self.cv_std           = data.get("cv_std", 0.0)
        self.holdout_baseline_home = data.get("holdout_baseline_home", 0.0)
        self.holdout_baseline_elo  = data.get("holdout_baseline_elo", 0.0)
        self._live_elo = self._live_accs = self._live_season = None
        td = self.trained_date.strftime("%Y-%m-%d") if self.trained_date else "?"
        print(f"  Team model loaded  (trained {td}, "
              f"holdout {self.cv_mean:.1%})")
        return self


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    from db_features import get_connection
    con = get_connection()
    print("Connected!\n")

    model = TeamModel()
    model.train(con)

    print("\nTesting prediction...")
    teams = con.execute("SELECT TeamID, TeamName FROM Teams LIMIT 2").fetchall()
    if len(teams) >= 2:
        pred = model.predict_game(con, teams[0][0], teams[1][0], "2025-26")
        if pred:
            print(f"\n  {pred['away_team']} @ {pred['home_team']}")
            print(f"  Winner: {pred['predicted_winner']} ({pred['confidence']:.1%})")
            print(f"  Score:  {pred['away_score']} - {pred['home_score']}")
            print(f"  OT:     {pred['ot_prob']:.1%}")
            print(f"  Elo edge: {pred['elo_diff']:+.0f}  "
                  f"(home Elo win prob {pred['elo_home_prob']:.1%})")

    con.close()
    print("\nDone!")
