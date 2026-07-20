"""
backfill_predictions.py
------------------------
Re-predict the existing 2025-26 Predictions rows with the NEW point-in-time
model, using WALK-FORWARD (each game predicted by a model trained only on
games strictly before it) so the backfilled accuracy is honest — the same
methodology that produced backtest.py's 54.7%, NOT an in-sample fit.

Safety:
  - Snapshots the current 2025-26 rows into Predictions_backup_prebackfill
    (created once; never overwritten) before any UPDATE, so this is reversible.
  - Only UPDATEs rows that already exist (does not insert new games).

Updates per row: PredictedWinner(+TeamID), PredictedHome/AwayScore,
ModelConfidencePct, PredictedOT, HomeIce/xGF/GSAX differentials, and
recomputes MLCorrect / ActualOutcomeCorrect against the stored actual result.

    python backfill_predictions.py            # do it
    python backfill_predictions.py --dry-run  # compute + report, write nothing
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from db_features import get_connection
from pit_features import load_games, build_feature_table
from team_model import TeamModel, W_RF, W_GB, W_LR

STEP_DAYS = 7
SEASON = "2025-26"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--step", type=int, default=STEP_DAYS)
    args = ap.parse_args()

    tm = TeamModel()
    con = get_connection()

    print("Loading games + building point-in-time features...")
    games = load_games(con)
    X, meta = build_feature_table(games)
    X = X.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    X_np = X.to_numpy()

    teams = {int(t): n for t, n in
             con.execute("SELECT TeamID, TeamName FROM Teams").fetchall()}

    # Existing rows to update: GameID -> ActualWinner
    existing = {}
    for gid, actual in con.execute(f"""
        SELECT GameID, ActualWinner FROM Predictions
        WHERE Season = '{SEASON}' AND GameID IS NOT NULL
          AND PredictedWinner IS NOT NULL
    """).fetchall():
        existing[int(gid)] = actual
    print(f"  Existing {SEASON} prediction rows: {len(existing)}")

    # Safety backup (once)
    if not args.dry_run:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS Predictions_backup_prebackfill AS
            SELECT * FROM Predictions WHERE Season = '{SEASON}'
        """)
        n_bak = con.execute(
            "SELECT COUNT(*) FROM Predictions_backup_prebackfill").fetchone()[0]
        print(f"  Backup table Predictions_backup_prebackfill: {n_bak} rows")

    y_w  = meta["y_home_win"].to_numpy()
    y_h  = meta["HomeScore"].to_numpy()
    y_a  = meta["AwayScore"].to_numpy()
    y_ot = meta["OvertimeFlag"].to_numpy()
    sw   = np.where(meta["IsPlayoff"].to_numpy() == 1, 0.5, 1.0)
    dates = pd.to_datetime(meta["GameDate"])

    season_mask = (meta["Season"] == SEASON).to_numpy()
    sidx = meta.index[season_mask]
    start, end = dates[sidx].min(), dates[sidx].max()

    updates = []
    cursor = start
    folds = 0
    while cursor <= end:
        hi = cursor + pd.Timedelta(days=args.step)
        block = meta.index[season_mask & (dates.to_numpy() >= np.datetime64(cursor))
                           & (dates.to_numpy() < np.datetime64(hi))]
        if len(block) == 0:
            cursor = hi
            continue
        train = meta.index[dates.to_numpy() < np.datetime64(cursor)]
        if len(train) < 100:
            cursor = hi
            continue

        scaler = StandardScaler().fit(X_np[train])
        Xtr = scaler.transform(X_np[train])
        Xte = scaler.transform(X_np[block])

        rf = tm._rf().fit(Xtr, y_w[train], sample_weight=sw[train])
        gb = tm._gb().fit(Xtr, y_w[train], sample_weight=sw[train])
        lr = tm._lr().fit(Xtr, y_w[train], sample_weight=sw[train])
        p = (W_RF * rf.predict_proba(Xte)[:, 1]
             + W_GB * gb.predict_proba(Xte)[:, 1]
             + W_LR * lr.predict_proba(Xte)[:, 1])

        hm = tm._pois().fit(Xtr, y_h[train], sample_weight=sw[train])
        am = tm._pois().fit(Xtr, y_a[train], sample_weight=sw[train])
        lam_h = np.clip(hm.predict(Xte), 0.5, 9)
        lam_a = np.clip(am.predict(Xte), 0.5, 9)

        otm = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=15, random_state=42
        ).fit(Xtr, y_ot[train], sample_weight=sw[train])
        ot_p = otm.predict_proba(Xte)[:, 1]

        for k, gi in enumerate(block):
            game_id = int(meta.at[gi, "GameID"])
            if game_id not in existing:
                continue
            ph = float(p[k])
            home = ph >= 0.5
            conf = ph if home else 1 - ph
            hid = int(meta.at[gi, "HomeTeamID"])
            aid = int(meta.at[gi, "AwayTeamID"])
            wid = hid if home else aid
            wname = teams[wid]
            is_ot = bool(ot_p[k] > 0.45)
            hs, as_ = tm._poisson_scoreline(float(lam_h[k]), float(lam_a[k]),
                                            home, is_ot)
            homeice = round((float(X.at[gi, "home_home_win_pct"])
                             - float(X.at[gi, "away_away_win_pct"])) * 6, 3)
            xgf  = round(float(X.at[gi, "goal_diff_pg_diff"]), 2)
            gsax = round(float(X.at[gi, "elo_diff"]) / 100.0, 2)

            actual = existing[game_id]
            mlc = None if actual is None else (1 if wname == actual else 0)
            updates.append({
                "game_id": game_id, "wid": wid, "wname": wname,
                "hs": hs, "as_": as_, "conf": round(conf * 100, 2),
                "is_ot": is_ot, "homeice": homeice, "xgf": xgf, "gsax": gsax,
                "mlc": mlc,
            })
        folds += 1
        cursor = hi

    # Report
    scored = [u for u in updates if u["mlc"] is not None]
    correct = sum(u["mlc"] for u in scored)
    print(f"\n  Folds: {folds}  |  rows to update: {len(updates)}  "
          f"|  completed (scored): {len(scored)}")
    if scored:
        print(f"  New walk-forward accuracy: {correct}/{len(scored)} "
              f"= {correct/len(scored):.1%}")

    if args.dry_run:
        print("\n  DRY RUN — no rows written.")
        con.close()
        return

    print("\n  Writing updates...")
    for u in updates:
        con.execute("""
            UPDATE Predictions SET
                PredictedWinnerTeamID = ?, PredictedWinner = ?,
                PredictedHomeScore = ?, PredictedAwayScore = ?,
                ModelConfidencePct = ?, PredictedOT = ?,
                HomeIceDifferential = ?, xGFDifferential = ?, GSAXDifferential = ?,
                MLCorrect = ?, ActualOutcomeCorrect = ?
            WHERE GameID = ? AND Season = ?
        """, [u["wid"], u["wname"], u["hs"], u["as_"], u["conf"], u["is_ot"],
              u["homeice"], u["xgf"], u["gsax"],
              u["mlc"], (None if u["mlc"] is None else bool(u["mlc"])),
              u["game_id"], SEASON])
    print(f"  Updated {len(updates)} rows.")

    # Verify against DB
    row = con.execute(f"""
        SELECT SUM(CASE WHEN MLCorrect=1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN MLCorrect IS NOT NULL THEN 1 ELSE 0 END)
        FROM Predictions WHERE Season='{SEASON}' AND PredictedWinner IS NOT NULL
    """).fetchone()
    if row and row[1]:
        print(f"  DB now shows: {row[0]}/{row[1]} = {row[0]/row[1]:.1%}")
    con.close()
    print("\nDONE")


if __name__ == "__main__":
    main()
