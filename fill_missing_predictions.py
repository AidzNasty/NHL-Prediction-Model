"""
fill_missing_predictions.py
----------------------------
Reusable utility to backfill predictions for COMPLETED games that have no
prediction rows yet — both the team game prediction (Predictions) and the
per-player props (PlayerPredictions).

Every prediction is generated WALK-FORWARD: the model is trained only on games
that finished strictly BEFORE the target game's date, so the backfilled numbers
are honest out-of-sample (leak-free), matching backtest.py / the dashboard.
Actual results are filled in from the database so rows show correct/wrong
immediately.

This complements backfill_predictions.py (which RE-SCORES existing rows). This
one INSERTS rows for games that were never predicted (e.g. a gap where the
daily pipeline didn't run).

Scope is explicit so it can't mass-insert by accident:
    python fill_missing_predictions.py --dry-run
        List completed games missing predictions and show what would be written.
    python fill_missing_predictions.py --game-ids 2790,2791,2792
        Fill only those games.
    python fill_missing_predictions.py --all-missing
        Fill EVERY completed game in the season that lacks a prediction row.
    python fill_missing_predictions.py --game-ids 2790,2791 --no-players
        Team predictions only (skip player props).

Options: --season (default 2025-26), --top-n (players per team, default 10).

Efficiency: games are grouped by date and the model is trained once per date
(all games on a date share "train on everything before this date"), so filling
many games stays cheap.
"""

import argparse
import numpy as np
import pandas as pd
from datetime import date

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from db_features import get_connection
from pit_features import load_games, build_feature_table
from team_model import TeamModel, W_RF, W_GB, W_LR
from pit_player_features import (
    load_player_games, prior_season_rates, build_opp_defense,
    compute_live_player_state, snapshot_player, _PlayerAcc,
)
from player_model import PlayerModel

GOAL_THRESH  = 0.20   # matches update_player_accuracy.py
POINT_THRESH = 0.40


def _ot_model():
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=15, random_state=42)


def find_missing_games(con, season, game_ids=None):
    """Completed games in `season` that have no Predictions row."""
    filt = ""
    if game_ids:
        filt = f" AND g.GameID IN ({','.join(map(str, game_ids))})"
    rows = con.execute(f"""
        SELECT g.GameID, g.GameDate, g.HomeTeamID, g.AwayTeamID,
               g.HomeScore, g.AwayScore, COALESCE(g.OvertimeFlag, FALSE),
               COALESCE(g.HomeIsBackToBack, FALSE), COALESCE(g.AwayIsBackToBack, FALSE),
               CASE WHEN g.GameType='Playoffs' THEN 1 ELSE 0 END
        FROM Games g
        WHERE g.Season = '{season}'
          AND g.HomeScore IS NOT NULL AND g.AwayScore IS NOT NULL
          AND g.GameID NOT IN (
              SELECT GameID FROM Predictions WHERE GameID IS NOT NULL)
          {filt}
        ORDER BY g.GameDate
    """).fetchall()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--game-ids", type=str, default=None,
                    help="comma-separated GameIDs to fill")
    ap.add_argument("--all-missing", action="store_true",
                    help="fill every completed game lacking a prediction")
    ap.add_argument("--no-players", action="store_true",
                    help="skip player-prop predictions")
    ap.add_argument("--season", type=str, default="2025-26")
    ap.add_argument("--top-n", type=int, default=10, help="players per team")
    args = ap.parse_args()

    game_ids = None
    if args.game_ids:
        game_ids = [int(x) for x in args.game_ids.split(",") if x.strip()]
    if not game_ids and not args.all_missing and not args.dry_run:
        ap.error("specify --game-ids, --all-missing, or --dry-run")

    con = get_connection()
    targets = find_missing_games(con, args.season, game_ids)
    if not targets:
        print("No completed games missing predictions for the given scope.")
        con.close()
        return
    print(f"Games missing predictions: {len(targets)}")
    if game_ids is None and not args.all_missing:
        # dry-run without scope: just list them
        for t in targets[:50]:
            print(f"  GameID {t[0]}  {str(t[1])[:10]}")
        if len(targets) > 50:
            print(f"  ... and {len(targets) - 50} more")
        print("\n(Use --game-ids or --all-missing to write; --dry-run shows detail.)")
        con.close()
        return

    # ── shared setup ────────────────────────────────────────────
    tm = TeamModel()
    games = load_games(con)
    X, meta = build_feature_table(games)
    X = X.reset_index(drop=True); meta = meta.reset_index(drop=True)
    X_np = X.to_numpy()
    dates = pd.to_datetime(meta["GameDate"]).to_numpy()
    gid_to_idx = {int(meta.at[i, "GameID"]): i for i in meta.index}
    teams = {int(t): n for t, n in con.execute("SELECT TeamID, TeamName FROM Teams").fetchall()}

    y_w = meta["y_home_win"].to_numpy(); y_h = meta["HomeScore"].to_numpy()
    y_a = meta["AwayScore"].to_numpy(); y_ot = meta["OvertimeFlag"].to_numpy()
    sw  = np.where(meta["IsPlayoff"].to_numpy() == 1, 0.5, 1.0)

    do_players = not args.no_players
    if do_players:
        pm = PlayerModel(); pm.load()
        plog = load_player_games(con)
        pprior = prior_season_rates(plog)
        opp_def = build_opp_defense(con)

    pred_id = con.execute("SELECT COALESCE(MAX(PredictionID),0) FROM Predictions").fetchone()[0]
    ppid    = con.execute("SELECT COALESCE(MAX(PlayerPredictionID),0) FROM PlayerPredictions").fetchone()[0]
    team_inserts, player_inserts = [], []

    # group targets by date (train once per date)
    by_date = {}
    for t in targets:
        by_date.setdefault(pd.Timestamp(t[1]).normalize(), []).append(t)

    for D in sorted(by_date):
        train = np.where(dates < np.datetime64(D))[0]
        if len(train) < 100:
            print(f"  [skip] {D.date()}: not enough prior games ({len(train)})")
            continue
        scaler = StandardScaler().fit(X_np[train])
        Xtr = scaler.transform(X_np[train])
        rf = tm._rf().fit(Xtr, y_w[train], sample_weight=sw[train])
        gb = tm._gb().fit(Xtr, y_w[train], sample_weight=sw[train])
        lr = tm._lr().fit(Xtr, y_w[train], sample_weight=sw[train])
        hm = tm._pois().fit(Xtr, y_h[train], sample_weight=sw[train])
        am = tm._pois().fit(Xtr, y_a[train], sample_weight=sw[train])
        otm = _ot_model().fit(Xtr, y_ot[train], sample_weight=sw[train])
        accs = compute_live_player_state(plog[plog["GameDate"] < D]) if do_players else None

        for (gid, gdate, hid, aid, ah, aa, aot, hb2b, ab2b, isplay) in by_date[D]:
            gid, hid, aid, ah, aa = int(gid), int(hid), int(aid), int(ah), int(aa)
            gi = gid_to_idx[gid]
            xte = scaler.transform(X_np[[gi]])
            ph = float((W_RF*rf.predict_proba(xte)[:, 1] + W_GB*gb.predict_proba(xte)[:, 1]
                        + W_LR*lr.predict_proba(xte)[:, 1])[0])
            home = ph >= 0.5
            lam_h = float(np.clip(hm.predict(xte)[0], 0.5, 9))
            lam_a = float(np.clip(am.predict(xte)[0], 0.5, 9))
            ot_prob = float(otm.predict_proba(xte)[0, 1]); is_ot = ot_prob > 0.45
            hs, as_ = tm._poisson_scoreline(lam_h, lam_a, home, is_ot)
            wid = hid if home else aid; wname = teams[wid]
            conf = ph if home else 1 - ph
            homeice = round((float(X.at[gi, "home_home_win_pct"])
                             - float(X.at[gi, "away_away_win_pct"])) * 6, 3)
            xgf  = round(float(X.at[gi, "goal_diff_pg_diff"]), 2)
            gsax = round(float(X.at[gi, "elo_diff"]) / 100.0, 2)
            actual_winner = teams[hid] if ah > aa else teams[aid]
            mlc = 1 if wname == actual_winner else 0
            pred_id += 1
            team_inserts.append([pred_id, gid, hid, aid, wid, wname, hs, as_,
                                 round(conf*100, 2), bool(is_ot), homeice, xgf, gsax,
                                 round(lam_h, 2), round(lam_a, 2), bool(hb2b), bool(ab2b),
                                 pd.Timestamp(gdate).to_pydatetime(), args.season,
                                 actual_winner, ah, aa, bool(aot), mlc, bool(mlc)])
            print(f"  G{gid} {pd.Timestamp(gdate).date()}: {wname} ({conf:.0%}) "
                  f"{as_}-{hs}{' OT' if is_ot else ''} | actual {actual_winner} "
                  f"-> {'OK' if mlc else 'X'}")

            if do_players:
                parts = plog[plog["GameID"] == gid]
                cand = []
                for r in parts.itertuples(index=False):
                    opp_ga = opp_def.get((int(r.OppTeamID), gid), 3.0)
                    acc = accs.get((r.Season, int(r.PlayerID)), _PlayerAcc())
                    pr = pprior.get(r.Season, {}).get(int(r.PlayerID))
                    feats = snapshot_player(acc, pr, opp_ga, bool(r.IsHome),
                                            bool(r.IsB2B), is_playoff=isplay)
                    pr_out = pm._predict_row(feats)
                    cand.append({"pid": int(r.PlayerID), "tid": int(r.TeamID),
                                 "is_home": bool(r.IsHome), "g": pr_out["goal"],
                                 "a": pr_out["assist"], "p": pr_out["point"],
                                 "ag": int(r.Goals), "aa": int(r.Assists), "ap": int(r.Points)})
                if cand:
                    cdf = pd.DataFrame(cand)
                    chosen = pd.concat([grp.sort_values("p", ascending=False).head(args.top_n)
                                        for _, grp in cdf.groupby("tid")])
                    for r in chosen.itertuples(index=False):
                        ppid += 1
                        gc = (1 if r.ag > 0 else 0) if r.g >= GOAL_THRESH else None
                        pc = (1 if r.ap > 0 else 0) if r.p >= POINT_THRESH else None
                        player_inserts.append([ppid, pred_id, r.pid, gid,
                                               round(r.g, 3), round(r.a, 3), round(r.p, 3),
                                               bool(r.is_home), pd.Timestamp(gdate).date(),
                                               r.ag, r.aa, r.ap, gc, pc, date.today()])

    print(f"\nTeam predictions to insert:   {len(team_inserts)}")
    print(f"Player predictions to insert: {len(player_inserts)}")

    if args.dry_run:
        print("\nDRY RUN — nothing inserted.")
        con.close()
        return

    for row in team_inserts:
        con.execute("""
            INSERT INTO Predictions (
                PredictionID, GameID, HomeTeamID, AwayTeamID,
                PredictedWinnerTeamID, PredictedWinner, PredictedHomeScore, PredictedAwayScore,
                ModelConfidencePct, PredictedOT, HomeIceDifferential, xGFDifferential, GSAXDifferential,
                HomeProjectedGoals, AwayProjectedGoals, HomeIsBackToBack, AwayIsBackToBack,
                PredictionDate, Season, ActualWinner, ActualHomeScore, ActualAwayScore, ActualOT,
                MLCorrect, ActualOutcomeCorrect
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, row)
    for row in player_inserts:
        con.execute("""
            INSERT INTO PlayerPredictions (
                PlayerPredictionID, PredictionID, PlayerID, GameID,
                GoalProbability, AssistProbability, PointProbability, IsHome, PredictionDate,
                ActualGoals, ActualAssists, ActualPoints, GoalCorrect, PointCorrect, UpdatedDate
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, row)
    print(f"\nInserted {len(team_inserts)} team + {len(player_inserts)} player predictions.")
    con.close()
    print("DONE")


if __name__ == "__main__":
    main()
