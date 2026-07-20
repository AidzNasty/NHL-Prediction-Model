"""
backtest.py
------------
Honest, leak-free evaluation of the game-winner model.

Why this exists:
  The old team_model reported ~55-58% via a RANDOM train/test split on
  season-snapshot features -> optimistic and not a forecast. Live accuracy
  was 49% (below the 52% always-home baseline). This harness does proper
  WALK-FORWARD (expanding-window) validation on point-in-time features so
  the number it prints is comparable to real production accuracy.

Method:
  - Features come from pit_features.build_feature_table (leak-free).
  - We evaluate the 2025-26 regular season.
  - Walking forward in time-ordered blocks, each block is predicted by a
    model trained ONLY on games strictly before it (2024-25 + earlier
    2025-26). The scaler is fit inside each fold.
  - Reported vs two baselines: always-home and Elo-only.

Metrics: accuracy, log-loss, Brier score, and a monthly breakdown.

    python backtest.py
    python backtest.py --step 14      # retrain every 14 days (default 7)
    python backtest.py --model gb     # gb | rf | logit  (default: ensemble)
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from db_features import get_connection
from pit_features import load_games, build_feature_table

TEST_SEASON = "2025-26"


def make_model(kind):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=15,
            max_features="sqrt", random_state=42, n_jobs=-1)
    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, min_samples_leaf=20, random_state=42)
    if kind == "logit":
        return LogisticRegression(C=0.3, max_iter=1000)
    raise ValueError(kind)


def fit_predict_proba(kind, X_tr, y_tr, X_te):
    """Fit scaler + model on train fold, return P(home win) for test fold."""
    scaler = StandardScaler().fit(X_tr)
    Xtr = scaler.transform(X_tr)
    Xte = scaler.transform(X_te)
    if kind == "ensemble":
        rf = make_model("rf").fit(Xtr, y_tr)
        gb = make_model("gb").fit(Xtr, y_tr)
        lr = make_model("logit").fit(Xtr, y_tr)
        p = (0.4 * rf.predict_proba(Xte)[:, 1]
             + 0.4 * gb.predict_proba(Xte)[:, 1]
             + 0.2 * lr.predict_proba(Xte)[:, 1])
        return p
    m = make_model(kind).fit(Xtr, y_tr)
    return m.predict_proba(Xte)[:, 1]


def walk_forward(X, meta, kind="ensemble", step_days=7, min_train=400):
    """
    Expanding-window walk-forward over the TEST_SEASON regular season.
    Returns a DataFrame of per-game predictions with truth + baselines.
    """
    meta = meta.reset_index(drop=True)
    X = X.reset_index(drop=True)

    test_mask = (meta["Season"] == TEST_SEASON) & (meta["IsPlayoff"] == 0)
    test_idx  = meta.index[test_mask]
    test_dates = meta.loc[test_idx, "GameDate"]

    start = test_dates.min()
    end   = test_dates.max()

    results = []
    cursor = start
    n_folds = 0
    while cursor <= end:
        block_hi = cursor + pd.Timedelta(days=step_days)
        block_idx = meta.index[
            test_mask
            & (meta["GameDate"] >= cursor)
            & (meta["GameDate"] < block_hi)
        ]
        if len(block_idx) == 0:
            cursor = block_hi
            continue

        train_idx = meta.index[meta["GameDate"] < cursor]
        if len(train_idx) < min_train:
            cursor = block_hi
            continue

        X_tr, y_tr = X.loc[train_idx], meta.loc[train_idx, "y_home_win"]
        X_te       = X.loc[block_idx]
        p_home     = fit_predict_proba(kind, X_tr, y_tr.values, X_te)

        blk = meta.loc[block_idx, ["GameID", "GameDate", "y_home_win"]].copy()
        blk["p_home"]      = p_home
        blk["pred_home"]   = (p_home >= 0.5).astype(int)
        blk["elo_p_home"]  = X.loc[block_idx, "elo_home_prob"].values
        blk["elo_pred"]    = (blk["elo_p_home"] >= 0.5).astype(int)
        results.append(blk)
        n_folds += 1
        cursor = block_hi

    out = pd.concat(results, ignore_index=True)
    out.attrs["n_folds"] = n_folds
    return out


def report(out, kind):
    y = out["y_home_win"].values
    p = np.clip(out["p_home"].values, 1e-6, 1 - 1e-6)

    model_acc   = accuracy_score(y, out["pred_home"])
    elo_acc     = accuracy_score(y, out["elo_pred"])
    home_acc    = y.mean()                      # always-home
    ll          = log_loss(y, p)
    brier       = brier_score_loss(y, p)
    ll_elo      = log_loss(y, np.clip(out["elo_p_home"], 1e-6, 1 - 1e-6))

    print("\n" + "=" * 60)
    print(f"WALK-FORWARD BACKTEST  —  {TEST_SEASON} regular season")
    print(f"model = {kind}   folds = {out.attrs.get('n_folds', '?')}   "
          f"games scored = {len(out)}")
    print("=" * 60)
    print(f"  Accuracy")
    print(f"    Always-home baseline : {home_acc:6.1%}")
    print(f"    Elo-only baseline    : {elo_acc:6.1%}")
    print(f"    >> MODEL             : {model_acc:6.1%}   "
          f"({model_acc - home_acc:+.1%} vs home, {model_acc - elo_acc:+.1%} vs Elo)")
    print(f"  Probabilistic quality (lower = better)")
    print(f"    Log-loss  model / elo: {ll:.4f} / {ll_elo:.4f}")
    print(f"    Brier     model      : {brier:.4f}")

    # Monthly breakdown
    out = out.copy()
    out["month"] = pd.to_datetime(out["GameDate"]).dt.to_period("M").astype(str)
    print(f"\n  By month:")
    print(f"    {'month':9} {'n':>4} {'home':>6} {'elo':>6} {'model':>6}")
    for m, grp in out.groupby("month"):
        ya = grp["y_home_win"].values
        print(f"    {m:9} {len(grp):>4} "
              f"{ya.mean():>6.1%} "
              f"{accuracy_score(ya, grp['elo_pred']):>6.1%} "
              f"{accuracy_score(ya, grp['pred_home']):>6.1%}")

    # Accuracy by model confidence — is the model well-calibrated on picks?
    print(f"\n  By model confidence (|p-0.5|):")
    out["conf"] = (out["p_home"] - 0.5).abs()
    out["correct"] = (out["pred_home"] == out["y_home_win"]).astype(int)
    bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50)]
    print(f"    {'band':12} {'n':>4} {'acc':>6}")
    for lo, hi in bins:
        grp = out[(out["conf"] >= lo) & (out["conf"] < hi)]
        if len(grp):
            print(f"    {f'{lo:.2f}-{hi:.2f}':12} {len(grp):>4} "
                  f"{grp['correct'].mean():>6.1%}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=7,
                    help="retrain cadence in days (default 7)")
    ap.add_argument("--model", type=str, default="ensemble",
                    choices=["ensemble", "rf", "gb", "logit"])
    ap.add_argument("--min-train", type=int, default=400)
    args = ap.parse_args()

    print("Connecting to MotherDuck...")
    con = get_connection()
    games = load_games(con)
    print(f"  {len(games)} completed games loaded")
    X, meta = build_feature_table(games)
    print(f"  Feature table: {X.shape[0]} x {X.shape[1]}")
    con.close()

    out = walk_forward(X, meta, kind=args.model,
                       step_days=args.step, min_train=args.min_train)
    report(out, args.model)


if __name__ == "__main__":
    main()
