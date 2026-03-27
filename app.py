"""
app.py — NHL Prediction Model Dashboard
Run: streamlit run app.py
"""

import os
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# Support both local .env and Streamlit Cloud secrets
def get_token():
    # Try Streamlit secrets first (deployed)
    try:
        return st.secrets["MOTHERDUCK_TOKEN"], st.secrets.get("MOTHERDUCK_DB", "my_db")
    except:
        pass
    # Fall back to .env (local)
    return os.getenv("MOTHERDUCK_TOKEN"), os.getenv("MOTHERDUCK_DB", "my_db")

TOKEN, DB = get_token()

# -- Page config -----------------------------------------------
st.set_page_config(
    page_title="NHL Predictions",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --cyan:  #00B4D8;
    --dark:  #0A0E1A;
    --card:  #111827;
    --border:#1F2937;
    --text:  #F9FAFB;
    --muted: #9CA3AF;
}

.stApp { background-color: var(--dark); color: var(--text); }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1628 0%, #0A0E1A 100%);
    border-right: 1px solid var(--border);
}

h1 { font-family: 'Bebas Neue', cursive !important;
     color: #00B4D8 !important; font-size: 2.8rem !important;
     letter-spacing: 3px !important; }
h2 { font-family: 'Bebas Neue', cursive !important;
     color: #F9FAFB !important; font-size: 1.6rem !important; }
h3 { font-family: 'Bebas Neue', cursive !important;
     color: #9CA3AF !important; font-size: 1.2rem !important; }

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 16px;
}
div[data-testid="stMetricValue"] { color: #00B4D8 !important; font-family: 'Bebas Neue', cursive !important; font-size: 2rem !important; }
div[data-testid="stMetricLabel"] { color: #9CA3AF !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }

div[data-testid="stDataFrame"] { background: #111827; border-radius: 8px; }

.stRadio > label { color: #9CA3AF !important; }
.stSelectbox > label { color: #9CA3AF !important; }
</style>
""", unsafe_allow_html=True)

# -- DB Connection ---------------------------------------------
def get_con():
    return duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")

@st.cache_data(ttl=300)
def q(sql):
    try:
        con = get_con()
        result = con.execute(sql).df()
        con.close()
        return result
    except Exception as e:
        st.error(f"DB error: {e}")
        return pd.DataFrame()

# Check if ActualGoals column exists in PlayerPredictions
def has_actual_cols():
    try:
        con = get_con()
        cols = [r[0].lower() for r in con.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'PlayerPredictions'
        """).fetchall()]
        con.close()
        return "actualgoals" in cols
    except:
        return False

HAS_ACTUALS = has_actual_cols()

# -- Sidebar ---------------------------------------------------
with st.sidebar:
    st.markdown("# 🏒 NHL PREDICTIONS")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Today's Games", "Player Props", "Model Accuracy", "Team Stats", "Stats Guide"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    try:
        _con = get_con()
        info = _con.execute("""
            SELECT
                MAX(PredictionDate)                                        AS last_run,
                COUNT(*)                                                   AS total,
                SUM(CASE WHEN ActualWinner IS NOT NULL THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN MLCorrect = 1 THEN 1 ELSE 0 END)            AS correct
            FROM Predictions WHERE Season = '2025-26'
        """).fetchone()
        _con.close()
        last_run, total, completed, correct = info
        acc = f"{correct/completed*100:.1f}%" if completed and completed > 0 else "—"
        st.markdown(f"""
        **Model Status**
        - Last run: `{str(last_run)[:10] if last_run else '—'}`
        - Season accuracy: `{acc}`
        - Predictions: `{total or 0}`
        """)
    except:
        st.markdown("*Model info unavailable*")

    st.markdown("---")
    st.caption("Data: Natural Stat Trick, NHL API\nModel: RF + GB Ensemble (60/40)\nCV Accuracy: 60.5% ±1.8%")

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — TODAY'S GAMES
# ═══════════════════════════════════════════════════════════════
if page == "Today's Games":
    st.title("TODAY'S GAMES")
    today = date.today()
    st.caption(today.strftime("%A, %B %d, %Y"))

    preds = q(f"""
        SELECT
            p.PredictionID, p.GameID,
            p.HomeTeamID, p.AwayTeamID,
            ht.TeamName  AS HomeTeam,
            awt.TeamName AS AwayTeam,
            p.PredictedWinner,
            COALESCE(p.PredictedHomeScore, 0) AS PredictedHomeScore,
            COALESCE(p.PredictedAwayScore, 0) AS PredictedAwayScore,
            COALESCE(p.ModelConfidencePct, 50) AS Confidence,
            p.PredictedOT,
            COALESCE(p.HomeIceDifferential, 0)  AS HomeIce,
            COALESCE(p.xGFDifferential, 0)       AS xGFDiff,
            COALESCE(p.GSAXDifferential, 0)      AS GSAXDiff,
            COALESCE(p.HomeProjectedGoals, 0)    AS HomeProj,
            COALESCE(p.AwayProjectedGoals, 0)    AS AwayProj,
            COALESCE(p.HomeIsBackToBack, FALSE)  AS HomB2B,
            COALESCE(p.AwayIsBackToBack, FALSE)  AS AwayB2B,
            p.ActualWinner,
            p.ActualHomeScore,
            p.ActualAwayScore,
            p.MLCorrect
        FROM Predictions p
        JOIN Teams ht  ON p.HomeTeamID = ht.TeamID
        JOIN Teams awt ON p.AwayTeamID = awt.TeamID
        WHERE CAST(p.PredictionDate AS DATE) = '{today}'
          AND p.PredictedWinner IS NOT NULL
        ORDER BY p.PredictionID
    """)

    if preds.empty:
        st.info("No predictions for today. Run `train_and_predict.py` to generate.")
    else:
        st.markdown(f"**{len(preds)} games predicted today**")
        st.divider()

        for _, pred in preds.iterrows():
            home    = str(pred["HomeTeam"])
            away    = str(pred["AwayTeam"])
            winner  = str(pred["PredictedWinner"])
            conf    = float(pred["Confidence"])
            h_score = int(pred["PredictedHomeScore"])
            a_score = int(pred["PredictedAwayScore"])
            is_ot   = bool(pred["PredictedOT"])
            homeice = float(pred["HomeIce"])
            xgf     = float(pred["xGFDiff"])
            gsax    = float(pred["GSAXDiff"])
            home_b2b= bool(pred["HomB2B"])
            away_b2b= bool(pred["AwayB2B"])
            home_proj= float(pred["HomeProj"])
            away_proj= float(pred["AwayProj"])
            game_id = int(pred["GameID"])

            # Card container
            with st.container(border=True):
                # Header row
                col_title, col_conf = st.columns([3, 1])
                with col_title:
                    st.markdown(
                        f"### {away} @ {home}",
                    )
                    if home_b2b:
                        st.markdown(f"🟠 **{home} on B2B**")
                    if away_b2b:
                        st.markdown(f"🟠 **{away} on B2B**")

                with col_conf:
                    conf_color = "🟢" if conf >= 65 else "🟡" if conf >= 55 else "🔴"
                    st.metric("Confidence", f"{conf:.1f}%")
                    if is_ot:
                        st.caption("⏱ OT likely")

                st.divider()

                # Score prediction
                col_away, col_vs, col_home = st.columns([2, 1, 2])
                with col_away:
                    away_color = "normal" if winner == away else "off"
                    st.metric(
                        away,
                        a_score,
                        delta=f"{away_proj:.1f} proj goals",
                        delta_color="off"
                    )
                    if winner == away:
                        st.markdown("🏆 **PREDICTED WINNER**")

                with col_vs:
                    st.markdown(
                        "<div style='text-align:center; padding-top:20px; color:#6B7280; font-size:1.5rem;'>@</div>",
                        unsafe_allow_html=True
                    )

                with col_home:
                    st.metric(
                        home,
                        h_score,
                        delta=f"{home_proj:.1f} proj goals",
                        delta_color="off"
                    )
                    if winner == home:
                        st.markdown("🏆 **PREDICTED WINNER**")

                st.divider()

                # Analytics row
                c1, c2, c3 = st.columns(3)
                with c1:
                    color = "normal" if homeice > 0 else "inverse"
                    st.metric("HomeIce Diff", f"{homeice:+.2f}", delta_color=color)
                with c2:
                    color = "normal" if xgf > 0 else "inverse"
                    st.metric("xGF% Diff", f"{xgf:+.1f}%", delta_color=color)
                with c3:
                    color = "normal" if gsax > 0 else "inverse"
                    st.metric("GSAX Diff", f"{gsax:+.1f}", delta_color=color)

                # Actual result if completed
                actual_w = pred["ActualWinner"]
                if actual_w and str(actual_w) not in ("nan", "None", ""):
                    ah = int(pred["ActualHomeScore"] or 0)
                    aa = int(pred["ActualAwayScore"] or 0)
                    correct = str(actual_w) == winner
                    icon = "✅ CORRECT" if correct else "❌ WRONG"
                    st.success(f"**Result:** {away} {aa} — {ah} {home} | Predicted: {winner} | {icon}")

                # Player predictions
                player_df = q(f"""
                    SELECT
                        pl.FirstName || ' ' || pl.LastName AS Player,
                        pl.Position AS Pos,
                        pp.GoalProbability   AS G,
                        pp.AssistProbability AS A,
                        pp.PointProbability  AS P,
                        pp.IsHome
                    FROM PlayerPredictions pp
                    JOIN Players pl ON pp.PlayerID = pl.PlayerID
                    WHERE pp.GameID = {game_id}
                      AND CAST(pp.PredictionDate AS DATE) = '{today}'
                    ORDER BY pp.PointProbability DESC
                    LIMIT 12
                """)

                if not player_df.empty:
                    st.markdown("**Top Player Projections**")
                    col_h, col_a = st.columns(2)

                    with col_h:
                        st.caption(f"🏠 {home}")
                        home_players = player_df[player_df["IsHome"] == True].head(6)
                        for _, p in home_players.iterrows():
                            g = float(p["G"] or 0)
                            a = float(p["A"] or 0)
                            pt= float(p["P"] or 0)
                            st.markdown(
                                f"**{p['Player']}** `{p['Pos']}`  "
                                f"🔴 G:{g:.0%}  🔵 A:{a:.0%}  🟢 P:{pt:.0%}"
                            )

                    with col_a:
                        st.caption(f"✈ {away}")
                        away_players = player_df[player_df["IsHome"] == False].head(6)
                        for _, p in away_players.iterrows():
                            g = float(p["G"] or 0)
                            a = float(p["A"] or 0)
                            pt= float(p["P"] or 0)
                            st.markdown(
                                f"**{p['Player']}** `{p['Pos']}`  "
                                f"🔴 G:{g:.0%}  🔵 A:{a:.0%}  🟢 P:{pt:.0%}"
                            )

            st.markdown("")

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — PLAYER PROPS
# ═══════════════════════════════════════════════════════════════
elif page == "Player Props":
    st.title("PLAYER PROPS")
    today = date.today()
    st.caption(f"Projections for {today.strftime('%B %d, %Y')}")

    # Build query based on whether actuals columns exist
    actual_cols = ", pp.ActualGoals, pp.ActualAssists, pp.ActualPoints" if HAS_ACTUALS else ""

    players = q(f"""
        SELECT
            pl.FirstName || ' ' || pl.LastName AS Player,
            pl.Position                        AS Pos,
            t.TeamName                         AS Team,
            pp.GoalProbability                 AS Goal_Prob,
            pp.AssistProbability               AS Assist_Prob,
            pp.PointProbability                AS Point_Prob,
            pp.IsHome
            {actual_cols}
        FROM PlayerPredictions pp
        JOIN Players pl ON pp.PlayerID = pl.PlayerID
        JOIN Teams t    ON pl.TeamID   = t.TeamID
        WHERE CAST(pp.PredictionDate AS DATE) = '{today}'
        ORDER BY pp.PointProbability DESC
    """)

    if players.empty:
        st.info("No player projections for today.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            teams = ["All Teams"] + sorted(players["Team"].unique().tolist())
            team_f = st.selectbox("Team", teams)
        with col2:
            pos_opts = ["All Positions"] + sorted(players["Pos"].dropna().unique().tolist())
            pos_f = st.selectbox("Position", pos_opts)
        with col3:
            min_p = st.slider("Min Point Probability", 0, 80, 20, 5)

        df = players.copy()
        if team_f != "All Teams":
            df = df[df["Team"] == team_f]
        if pos_f != "All Positions":
            df = df[df["Pos"] == pos_f]
        df = df[df["Point_Prob"] >= min_p / 100]

        st.markdown(f"**{len(df)} players** matching filters")
        st.divider()

        # Format display
        disp = df.copy()
        disp["Goal %"]   = disp["Goal_Prob"].apply(lambda x: f"{float(x):.0%}" if pd.notna(x) else "—")
        disp["Assist %"] = disp["Assist_Prob"].apply(lambda x: f"{float(x):.0%}" if pd.notna(x) else "—")
        disp["Point %"]  = disp["Point_Prob"].apply(lambda x: f"{float(x):.0%}" if pd.notna(x) else "—")
        disp["Location"] = disp["IsHome"].apply(lambda x: "🏠 Home" if x else "✈ Away")

        show = ["Player", "Pos", "Team", "Location", "Goal %", "Assist %", "Point %"]

        if HAS_ACTUALS and "ActualPoints" in disp.columns:
            disp["Actual"] = disp["ActualPoints"].apply(
                lambda x: f"{int(x)} pts" if pd.notna(x) else "—"
            )
            show.append("Actual")

        st.dataframe(disp[show], use_container_width=True, hide_index=True)

        # Chart
        st.markdown("### TOP 20 BY POINT PROBABILITY")
        top20 = df.head(20).copy()
        fig = go.Figure()
        fig.add_bar(x=top20["Player"], y=top20["Goal_Prob"],   name="Goal",   marker_color="#EF4444")
        fig.add_bar(x=top20["Player"], y=top20["Assist_Prob"], name="Assist", marker_color="#3B82F6")
        fig.update_layout(
            barmode="stack",
            paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
            font=dict(color="#F9FAFB"),
            legend=dict(bgcolor="#111827"),
            xaxis=dict(tickangle=45, gridcolor="#1F2937"),
            yaxis=dict(gridcolor="#1F2937", tickformat=".0%", title="Probability"),
            height=420, margin=dict(t=20, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ACCURACY
# ═══════════════════════════════════════════════════════════════
elif page == "Model Accuracy":
    st.title("MODEL ACCURACY")

    try:
        _con2 = get_con()
        stats = _con2.execute("""
            SELECT
                COUNT(*)                                                AS total,
                SUM(CASE WHEN ActualWinner IS NOT NULL THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN MLCorrect = 1             THEN 1 ELSE 0 END) AS correct,
                SUM(CASE WHEN ModelConfidencePct >= 65
                     AND MLCorrect = 1             THEN 1 ELSE 0 END)  AS high_c,
                SUM(CASE WHEN ModelConfidencePct >= 65
                     AND ActualWinner IS NOT NULL  THEN 1 ELSE 0 END)  AS high_t,
                SUM(CASE WHEN ModelConfidencePct BETWEEN 55 AND 64.9
                     AND MLCorrect = 1             THEN 1 ELSE 0 END)  AS med_c,
                SUM(CASE WHEN ModelConfidencePct BETWEEN 55 AND 64.9
                     AND ActualWinner IS NOT NULL  THEN 1 ELSE 0 END)  AS med_t,
                SUM(CASE WHEN ModelConfidencePct < 55
                     AND MLCorrect = 1             THEN 1 ELSE 0 END)  AS low_c,
                SUM(CASE WHEN ModelConfidencePct < 55
                     AND ActualWinner IS NOT NULL  THEN 1 ELSE 0 END)  AS low_t
            FROM Predictions
            WHERE Season = '2025-26' AND PredictedWinner IS NOT NULL
        """).fetchone()
        _con2.close()

        total     = int(stats[0] or 0)
        completed = int(stats[1] or 0)
        correct   = int(stats[2] or 0)
        acc       = correct / completed * 100 if completed > 0 else 0.0
        high_c, high_t = int(stats[3] or 0), int(stats[4] or 0)
        med_c,  med_t  = int(stats[5] or 0), int(stats[6] or 0)
        low_c,  low_t  = int(stats[7] or 0), int(stats[8] or 0)

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Season Accuracy", f"{acc:.1f}%" if completed > 0 else "—")
        c2.metric("Correct / Completed", f"{correct} / {completed}")
        c3.metric("CV Accuracy (5-fold)", "60.5% ±1.8%")
        c4.metric("Total Predictions", total)

        st.divider()

        # Accuracy by tier
        st.markdown("### ACCURACY BY CONFIDENCE TIER")
        tiers = [
            {"Tier": "High (>=65%)",    "Acc": high_c/high_t*100 if high_t > 0 else 0, "N": high_t},
            {"Tier": "Medium (55-65%)", "Acc": med_c/med_t*100  if med_t  > 0 else 0, "N": med_t},
            {"Tier": "Low (<55%)",      "Acc": low_c/low_t*100  if low_t  > 0 else 0, "N": low_t},
        ]
        tier_df = pd.DataFrame(tiers)
        fig2 = px.bar(
            tier_df, x="Tier", y="Acc",
            text=tier_df.apply(
                lambda r: f"{r['Acc']:.1f}% ({int(r['N'])} games)" if r["N"] > 0 else "No data",
                axis=1
            ),
            color="Acc",
            color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
            range_color=[45, 75],
        )
        fig2.add_hline(y=50, line_dash="dot",  line_color="#6B7280", annotation_text="Coin flip")
        fig2.add_hline(y=60.5, line_dash="dash", line_color="#00B4D8", annotation_text="CV baseline")
        fig2.update_layout(
            paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
            font=dict(color="#F9FAFB"),
            yaxis=dict(range=[0, 85], gridcolor="#1F2937", title="Accuracy %"),
            xaxis=dict(gridcolor="#1F2937"),
            height=320, margin=dict(t=20),
            coloraxis_showscale=False,
            showlegend=False,
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

        # Recent predictions
        st.markdown("### RECENT PREDICTIONS")
        recent = q("""
            SELECT
                CAST(p.PredictionDate AS DATE)                              AS Date,
                awt.TeamName                                                AS Away,
                ht.TeamName                                                 AS Home,
                p.PredictedWinner                                           AS Predicted,
                ROUND(COALESCE(p.ModelConfidencePct, 0), 1)                AS "Conf %",
                p.ActualWinner                                              AS Actual,
                CASE WHEN p.ActualWinner IS NULL THEN 'Pending'
                     WHEN p.MLCorrect = 1        THEN '✅ Correct'
                     ELSE '❌ Wrong' END                                    AS Result
            FROM Predictions p
            JOIN Teams ht  ON p.HomeTeamID = ht.TeamID
            JOIN Teams awt ON p.AwayTeamID = awt.TeamID
            WHERE p.Season = '2025-26' AND p.PredictedWinner IS NOT NULL
            ORDER BY p.PredictionDate DESC
            LIMIT 25
        """)
        if not recent.empty:
            st.dataframe(recent, use_container_width=True, hide_index=True)

        # Rolling accuracy (only if we have completed games)
        if completed >= 5:
            st.markdown("### ROLLING 10-GAME ACCURACY")
            rolling = q("""
                SELECT CAST(PredictionDate AS DATE) AS Date, MLCorrect
                FROM Predictions
                WHERE Season = '2025-26' AND ActualWinner IS NOT NULL
                  AND MLCorrect IS NOT NULL
                ORDER BY PredictionDate
            """)
            if len(rolling) >= 5:
                rolling["Rolling"] = rolling["MLCorrect"].rolling(10, min_periods=1).mean() * 100
                fig3 = px.line(rolling, x="Date", y="Rolling", line_shape="spline")
                fig3.add_hline(y=60.5, line_dash="dash", line_color="#00B4D8",
                               annotation_text="CV 60.5%")
                fig3.add_hline(y=50, line_dash="dot", line_color="#6B7280",
                               annotation_text="Coin flip")
                fig3.update_traces(line_color="#00B4D8", line_width=2)
                fig3.update_layout(
                    paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
                    font=dict(color="#F9FAFB"),
                    yaxis=dict(range=[30, 90], gridcolor="#1F2937", title="Accuracy %"),
                    xaxis=dict(gridcolor="#1F2937"),
                    height=320, margin=dict(t=20),
                )
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info(f"Rolling accuracy chart will appear once {max(0, 5-completed)} more games complete.")

    except Exception as e:
        st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — TEAM STATS
# ═══════════════════════════════════════════════════════════════
elif page == "Team Stats":
    st.title("TEAM STATS")

    season = st.selectbox("Season", ["2025-26", "2024-25"])

    teams_df = q(f"""
        SELECT
            t.TeamName                                          AS Team,
            ts.GP, ts.W, ts.L, ts.OTL, ts.Points,
            ROUND(ts.Point_Pct * 100, 1)                       AS "Point%",
            ROUND(ts.CF_Pct,   1)                              AS "CF%",
            ROUND(ts.FF_Pct,   1)                              AS "FF%",
            ROUND(ts.xGF_Pct,  1)                              AS "xGF%",
            ROUND(ts.HDCF_Pct, 1)                              AS "HDCF%",
            ROUND(ts.PDO,      1)                              AS PDO,
            ROUND(ts.SV_Pct,        2)                         AS "SV%",
            ROUND(ts.SH_Pct,   1)                              AS "SH%",
            ROUND(ts.GF * 1.0 / NULLIF(ts.GP,0), 2)           AS "GF/G",
            ROUND(ts.GA * 1.0 / NULLIF(ts.GP,0), 2)           AS "GA/G",
            ROUND(ts.HomeWinPct * 100, 1)                     AS "Home W%",
            ROUND(ts.AwayWinPct * 100, 1)                     AS "Away W%",
            ROUND(ts.Team_FaceoffWinPct, 1)                   AS "FO%"
        FROM TeamStandings ts
        JOIN Teams t ON ts.TeamID = t.TeamID
        WHERE ts.Season = '{season}' AND ts.GP > 0
        ORDER BY ts.Points DESC
    """)

    if teams_df.empty:
        st.info("No team stats available.")
    else:
        view = st.radio(
            "View",
            ["Standings", "Possession", "Scoring", "Home/Away"],
            horizontal=True
        )

        cols_map = {
            "Standings":  ["Team", "GP", "W", "L", "OTL", "Points", "Point%"],
            "Possession": ["Team", "CF%", "FF%", "xGF%", "HDCF%", "PDO"],
            "Scoring":    ["Team", "GF/G", "GA/G", "SH%", "SV%", "FO%"],
            "Home/Away":  ["Team", "Home W%", "Away W%", "GF/G", "GA/G"],
        }

        st.dataframe(
            teams_df[cols_map[view]],
            use_container_width=True,
            hide_index=True,
            height=580,
        )

        # Scatter plot
        st.markdown("### xGF% vs HDCF% — POSSESSION QUALITY")
        fig4 = px.scatter(
            teams_df, x="xGF%", y="HDCF%",
            text="Team", color="Points",
            color_continuous_scale="Blues",
            size="Points",
            hover_data=["GP", "W", "Points"],
        )
        fig4.add_vline(x=50, line_dash="dash", line_color="#374151")
        fig4.add_hline(y=50, line_dash="dash", line_color="#374151")
        fig4.update_traces(textposition="top center", textfont_size=9)
        fig4.update_layout(
            paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
            font=dict(color="#F9FAFB"),
            xaxis=dict(title="xGF%", gridcolor="#1F2937"),
            yaxis=dict(title="HDCF%", gridcolor="#1F2937"),
            height=520, margin=dict(t=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — STATS GUIDE
# ═══════════════════════════════════════════════════════════════
elif page == "Stats Guide":
    st.title("STATS GUIDE")
    st.caption("Definitions for every stat used in this model and dashboard")

    tab1, tab2, tab3 = st.tabs(["Possession & Shot Quality", "Goalie & Shooting", "Model Inputs"])

    with tab1:
        st.markdown("### Possession & Shot Quality Stats")
        st.markdown("*All possession stats are measured at 5-on-5 unless noted. Source: Natural Stat Trick.*")
        st.divider()

        stats = [
            ("CF% — Corsi For Percentage",
             "The percentage of all shot attempts (goals, shots on goal, missed shots, and blocked shots) taken by a team while that team is on the ice at 5-on-5.",
             "CF% = Shot Attempts For / (Shot Attempts For + Shot Attempts Against)",
             "A CF% above 50% means the team generates more shot attempts than they allow. It is the broadest measure of puck possession. League average is 50%."),

            ("FF% — Fenwick For Percentage",
             "Same as CF% but excludes blocked shots. Unblocked shot attempts (goals + shots on goal + missed shots) are considered a better proxy for shot quality than raw Corsi because blocked shots are partly a function of the opposing team's defense.",
             "FF% = Unblocked Attempts For / (Unblocked Attempts For + Unblocked Attempts Against)",
             "Generally tracks closely with CF%, but teams with shot-blocking defenses will show a bigger gap between their opponents' CF% and FF%."),

            ("xGF% — Expected Goals For Percentage",
             "The percentage of expected goals generated by a team, where each shot is weighted by its probability of being a goal based on shot type, location, and game situation.",
             "xGF% = xGoals For / (xGoals For + xGoals Against)",
             "This is the most predictive of the possession stats. A team can out-Corsi their opponent but still lose in xGF% if their shots come from bad angles. xGF% above 50% signals genuine shot-quality dominance. This is a primary input to the prediction model."),

            ("HDCF% — High-Danger Corsi For Percentage",
             "Corsi percentage restricted to only 'high-danger' shot attempts — shots from the slot and the area directly in front of the net, roughly within 20 feet of center ice.",
             "HDCF% = High-Danger Attempts For / (High-Danger Attempts For + High-Danger Attempts Against)",
             "High-danger chances convert to goals at a much higher rate than perimeter shots. A team dominating HDCF% is generating the most dangerous looks in the game. Pairs with xGF% to tell the full possession story."),

            ("PDO",
             "The sum of a team's 5-on-5 save percentage and shooting percentage. Named after a user on the old Hockey's Future forums.",
             "PDO = SV% + SH%  (e.g. 0.923 + 0.080 = 1.003, displayed as 100.3)",
             "PDO regresses strongly toward 100 over a full season. Teams above 102 are likely getting lucky; teams below 98 are likely due for better results. It is used in the model as a luck/sustainability signal."),
        ]

        for name, what, formula, interpretation in stats:
            with st.expander(name, expanded=False):
                st.markdown(f"**What it measures:** {what}")
                st.code(formula, language=None)
                st.markdown(f"**How to read it:** {interpretation}")

    with tab2:
        st.markdown("### Goalie & Shooting Stats")
        st.divider()

        stats2 = [
            ("SV% — Save Percentage",
             "The fraction of shots on goal that a goalie stops.",
             "SV% = Saves / Shots On Goal Against  (e.g. 0.921 = 92.1%)",
             "League average is typically around 0.906–0.912. A full-season SV% above 0.920 is elite. In this dashboard, SV% is shown as a percentage (e.g. 92.1)."),

            ("SH% — Shooting Percentage",
             "The percentage of shots on goal that become goals for the team's skaters.",
             "SH% = Goals / Shots On Goal  (e.g. 10.5 means 10.5%)",
             "League average is roughly 8–10%. Like PDO, team-level SH% regresses toward the mean. Sustained high SH% usually indicates elite skaters or good puck luck."),

            ("GSAX — Goals Saved Above Expected",
             "How many more goals a goalie prevented compared to what an average goalie would have allowed given the same shot profile. Accounts for shot location, type, and danger level.",
             "GSAX = Expected Goals Against - Actual Goals Against",
             "Positive GSAX = goalie outperformed expectations; negative = underperformed. This is the model's primary goalie-quality input. A full-season GSAX of +10 or more is an elite performance."),

            ("GF/G — Goals For Per Game",
             "Average number of goals scored per game across all situations (including power play and shorthanded).",
             "GF/G = Total Goals For / Games Played",
             "Reflects overall offensive output. Top teams typically sit above 3.5 GF/G. Pair with GA/G to assess net goal differential."),

            ("GA/G — Goals Against Per Game",
             "Average number of goals allowed per game.",
             "GA/G = Total Goals Against / Games Played",
             "Elite defenses/goaltending combinations allow under 2.5 GA/G. A team's GF/G minus GA/G gives their goal differential, one of the strongest predictors of playoff success."),

            ("FO% — Faceoff Win Percentage",
             "The percentage of faceoffs won by the team.",
             "FO% = Faceoffs Won / Total Faceoffs",
             "League average is 50%. Winning faceoffs gives a team possession to start shifts — especially valuable in the defensive zone and on draws after icing. Effect on outcome is modest but real."),
        ]

        for name, what, formula, interpretation in stats2:
            with st.expander(name, expanded=False):
                st.markdown(f"**What it measures:** {what}")
                st.code(formula, language=None)
                st.markdown(f"**How to read it:** {interpretation}")

    with tab3:
        st.markdown("### Model-Specific Inputs")
        st.caption("These are the differential metrics the model uses to compare the two teams in a given matchup.")
        st.divider()

        stats3 = [
            ("xGF Differential",
             "The difference between the home team's xGF% and the away team's xGF% over the last 10 games.",
             "xGF Diff = Home xGF% (rolling) - Away xGF% (rolling)",
             "Positive = home team has been the better possession team recently. One of the highest-weight features in the model."),

            ("GSAX Differential",
             "The difference in Goals Saved Above Expected between the home team's goalie and the away team's goalie.",
             "GSAX Diff = Home GSAX - Away GSAX",
             "Captures goalie matchup quality. A large positive value means the home team has a significant goaltending edge."),

            ("HomeIce Differential",
             "A composite home-ice advantage score that combines the home team's historical home win rate with league-average home-ice advantage.",
             "Calculated from team-specific home W% vs league baseline",
             "Some teams have significantly stronger home-ice advantage than others (e.g. loud arenas, altitude). This feature adjusts the model's baseline for each matchup."),

            ("Back-to-Back Flag",
             "Whether the home or away team is playing their second game in two nights.",
             "B2B = 1 if team played a game the previous calendar day, else 0",
             "Playing B2B meaningfully reduces win probability, especially for the team that also traveled. The model applies separate B2B penalties for home and away teams."),

            ("Model Confidence %",
             "The ensemble model's estimated probability that the predicted winner actually wins, expressed as a percentage.",
             "Confidence = max(P(home win), P(away win)) from the RF + GB ensemble",
             ">=65% = High confidence (green). 55-65% = Medium (yellow). <55% = Low (red). Historical CV accuracy is 60.5%."),
        ]

        for name, what, formula, interpretation in stats3:
            with st.expander(name, expanded=False):
                st.markdown(f"**What it measures:** {what}")
                st.code(formula, language=None)
                st.markdown(f"**How to read it:** {interpretation}")

        st.divider()
        st.markdown("### Data Sources")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Natural Stat Trick**
            - CF%, FF%, xGF%, HDCF%
            - GSAX, PDO
            - All 5-on-5 splits
            """)
        with col2:
            st.markdown("""
            **NHL Official API**
            - Game schedules & scores
            - Roster & player data
            - Starting goalies & lineups
            """)
        with col3:
            st.markdown("""
            **Hockey Reference**
            - Historical game stats
            - Team standings
            - Season-level aggregates
            """)
