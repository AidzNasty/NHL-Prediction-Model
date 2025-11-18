#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHL Prediction Model - Web App Version with ML Integration
Accessible via web browser - easy to share!
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="NHL Prediction Model 2025-26",
    page_icon="🏒",
    layout="wide"
)

# Custom CSS - Modern, Neutral Design - FULL WIDTH
st.markdown("""
    <style>
    /* Main App Styling - Neutral Colors */
    .main {
        background: #f5f5f5;
        padding: 2rem 1rem;
    }
    .stApp {
        background: #f5f5f5;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #2c3e50 !important;
        text-align: center;
    }
    h1 {
        margin-bottom: 0.5rem;
    }
    
    /* Full Width Container */
    .centered-container {
        max-width: 100%;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Game Card - Neutral Design */
    .game-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .game-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border-color: #d0d0d0;
    }
    
    /* Winner Styling - Good Green */
    .winner {
        color: #2e7d32;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 0.75rem;
        background: #e8f5e9;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid #4caf50;
    }
    .ml-winner {
        color: #2e7d32;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 0.75rem;
        background: #e8f5e9;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid #4caf50;
    }
    
    /* Metric Card - Neutral */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    /* Prediction Box */
    .prediction-box {
        background: #fafafa;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #9e9e9e;
        margin: 1rem 0;
    }
    
    /* Team Display */
    .team-display {
        text-align: center;
        padding: 1rem;
    }
    
    /* Probability Badge */
    .prob-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Differential Badge */
    .diff-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .game-card {
            padding: 1.5rem;
        }
        h1 {
            font-size: 1.75rem;
        }
        h2 {
            font-size: 1.5rem;
        }
        .centered-container {
            padding: 0 0.5rem;
        }
    }
    
    /* Spacing Improvements */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Sidebar Improvements - Neutral */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Streamlit default text color */
    .stMarkdown p, .stMarkdown li {
        color: #333333;
    }
    
    /* Better table styling */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    
    /* Correct/Incorrect highlighting */
    .correct-prediction {
        color: #2e7d32;
        font-weight: 600;
    }
    .incorrect-prediction {
        color: #c62828;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
EXCEL_FILE = 'Aidan Conte NHL 2025-26 Prediction Model.xlsx'
LEAGUE_AVG_TOTAL = 6.24
TEAM_WEIGHT = 0.70
HOMEICE_WEIGHT = 0.30

def get_probability_color(probability):
    """
    Convert probability (0-1) to color from red (weak) to green (strong)
    Returns hex color and background color
    """
    # Normalize probability to 0-1 range (assuming 0.5-1.0 is the range)
    normalized = (probability - 0.5) / 0.5  # Maps 0.5->0, 1.0->1
    normalized = max(0, min(1, normalized))  # Clamp to 0-1
    
    # Red to Green gradient
    if normalized < 0.5:
        # Red to Yellow
        t = normalized * 2
        r = int(211 + (251 - 211) * t)
        g = int(47 + (192 - 47) * t)
        b = int(47 + (45 - 47) * t)
    else:
        # Yellow to Green
        t = (normalized - 0.5) * 2
        r = int(251 + (56 - 251) * t)
        g = int(192 + (142 - 192) * t)
        b = int(45 + (60 - 45) * t)
    
    color = f"#{r:02x}{g:02x}{b:02x}"
    bg_color = f"rgba({r}, {g}, {b}, 0.15)"
    return color, bg_color

def get_differential_color(differential):
    """
    Convert homeice differential to color from red (negative/weak) to green (positive/strong)
    Returns hex color and background color
    """
    max_diff = 3.0
    normalized = (differential + max_diff) / (2 * max_diff)  # Maps -3->0, +3->1
    normalized = max(0, min(1, normalized))  # Clamp to 0-1
    
    # Red to Green gradient
    if normalized < 0.5:
        # Red to Yellow
        t = normalized * 2
        r = int(211 + (251 - 211) * t)
        g = int(47 + (192 - 47) * t)
        b = int(47 + (45 - 47) * t)
    else:
        # Yellow to Green
        t = (normalized - 0.5) * 2
        r = int(251 + (56 - 251) * t)
        g = int(192 + (142 - 192) * t)
        b = int(45 + (60 - 45) * t)
    
    color = f"#{r:02x}{g:02x}{b:02x}"
    bg_color = f"rgba({r}, {g}, {b}, 0.15)"
    return color, bg_color

@st.cache_data(ttl=3600)  # Cache expires after 1 hour (3600 seconds)
def load_data():
    """Load Excel data with caching"""
    load_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        predictions = pd.read_excel(EXCEL_FILE, sheet_name='NHL HomeIce Model', header=0)
        predictions = predictions.iloc[:, 3:].reset_index(drop=True)
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        # Try to load ML predictions (optional)
        ml_predictions = None
        try:
            ml_predictions = pd.read_excel(EXCEL_FILE, sheet_name='ML Prediction Model', header=0)
            ml_predictions['date'] = pd.to_datetime(ml_predictions['date'])
            st.sidebar.success("✅ ML Model loaded")
        except Exception as ml_error:
            st.sidebar.warning("⚠️ ML Model not available")
            st.sidebar.caption(f"Reason: {str(ml_error)}")
            ml_predictions = pd.DataFrame()  # Empty dataframe
        
        return predictions, standings, ml_predictions, load_timestamp
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {e}")
        st.info("Make sure 'Aidan Conte NHL 2025-26 Prediction Model.xlsx' is in the same folder as this script")
        return None, None, None, None

def calculate_prediction(home_team, away_team, standings):
    """Calculate prediction for a matchup"""
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    # Calculate HomeIce Differential
    home_home_win_pct = home_row['HomeWin%']
    away_away_win_pct = away_row['AwayWin%']
    homeice_diff = (home_home_win_pct - away_away_win_pct) * 6
    
    # Get goal stats
    home_offense = home_row.iloc[14]
    home_defense = home_row.iloc[16]
    away_offense = away_row.iloc[15]
    away_defense = away_row.iloc[17]
    
    # Method 1: Team-based prediction
    team_predicted_home = (home_offense + away_defense) / 2
    team_predicted_away = (away_offense + home_defense) / 2
    
    # Method 2: HomeIce Differential-based prediction
    avg_per_team = LEAGUE_AVG_TOTAL / 2
    homeice_predicted_home = avg_per_team + (homeice_diff / 2)
    homeice_predicted_away = avg_per_team - (homeice_diff / 2)
    
    # Blended prediction
    predicted_home_raw = (team_predicted_home * TEAM_WEIGHT) + (homeice_predicted_home * HOMEICE_WEIGHT)
    predicted_away_raw = (team_predicted_away * TEAM_WEIGHT) + (homeice_predicted_away * HOMEICE_WEIGHT)
    
    predicted_home = round(predicted_home_raw)
    predicted_away = round(predicted_away_raw)
    
    # Determine winner
    if predicted_home > predicted_away:
        predicted_winner = home_team
        win_prob = 0.5 + (abs(homeice_diff) / 12)
    elif predicted_away > predicted_home:
        predicted_winner = away_team
        win_prob = 0.5 + (abs(homeice_diff) / 12)
    else:
        if homeice_diff > 0:
            predicted_winner = home_team
            predicted_home += 1
        else:
            predicted_winner = away_team
            predicted_away += 1
        win_prob = 0.5 + (abs(homeice_diff) / 12)
    
    win_prob = min(0.85, max(0.52, win_prob))
    
    return {
        'predicted_winner': predicted_winner,
        'win_prob': win_prob,
        'predicted_home': predicted_home,
        'predicted_away': predicted_away,
        'homeice_diff': homeice_diff,
        'home_row': home_row,
        'away_row': away_row
    }

def get_ml_prediction(home_team, away_team, game_date, ml_predictions):
    """Get ML prediction for a specific game"""
    # Check if ML predictions are available
    if ml_predictions is None or len(ml_predictions) == 0:
        return {'has_ml': False}
    
    # Convert game_date to datetime if it's not already
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    
    # Find matching game in ML predictions
    ml_game = ml_predictions[
        (ml_predictions['home_team'] == home_team) & 
        (ml_predictions['away_team'] == away_team) &
        (ml_predictions['date'].dt.date == game_date.date())
    ]
    
    if len(ml_game) > 0:
        ml_game = ml_game.iloc[0]
        return {
            'ml_predicted_winner': ml_game['ml_predicted_winner'],
            'ml_home_win_prob': ml_game['ml_home_win_prob'],
            'ml_away_win_prob': ml_game['ml_away_win_prob'],
            'ml_confidence': ml_game['ml_confidence'],
            'ml_predicted_home': ml_game['ml_predicted_home_score'],
            'ml_predicted_away': ml_game['ml_predicted_away_score'],
            'has_ml': True
        }
    else:
        return {'has_ml': False}

def display_game_card(game, standings, ml_predictions):
    """Display a game card with both model predictions - Modern Layout"""
    home_team = game['Home']
    away_team = game['Visitor']
    game_time = game['Time']
    game_date = game['Date']
    
    # Get predictions from both models
    excel_prediction = calculate_prediction(home_team, away_team, standings)
    ml_prediction = get_ml_prediction(home_team, away_team, game_date, ml_predictions)
    
    home_row = excel_prediction['home_row']
    away_row = excel_prediction['away_row']
    
    # Create card with container
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Header with time - centered
        st.markdown(f"<div style='text-align: center; color: #666666; font-size: 1.1rem; margin-bottom: 1.5rem;'>🕐 {game_time}</div>", unsafe_allow_html=True)
        
        # Teams display - centered and responsive
        col1, col_vs, col2 = st.columns([2.5, 0.5, 2.5], gap="medium")
        
        with col1:
            st.markdown(f"<div class='team-display'>", unsafe_allow_html=True)
            st.markdown(f"### {away_team}")
            st.caption(f"**{int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])}** ({int(away_row['PTS'])} pts)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_vs:
            st.markdown("<div style='text-align: center; padding-top: 1.5rem; color: #666666;'><h2>@</h2></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div class='team-display'>", unsafe_allow_html=True)
            st.markdown(f"### {home_team}")
            st.caption(f"**{int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['OTL'])}** ({int(home_row['PTS'])} pts)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Get colors for probability and differential
        prob_color, prob_bg = get_probability_color(excel_prediction['win_prob'])
        diff_color, diff_bg = get_differential_color(excel_prediction['homeice_diff'])
        
        # Predictions side by side - responsive columns
        col_excel, col_ml = st.columns(2, gap="large")
        
        with col_excel:
            with st.container():
                st.markdown("#### 📊 Excel Model")
                st.markdown(f'<div class="winner">🏆 {excel_prediction["predicted_winner"]}</div>', unsafe_allow_html=True)
                
                # Win Probability with color
                st.markdown(f"**Win Probability:**")
                st.markdown(f'<span class="prob-badge" style="color: {prob_color}; background: {prob_bg};">{excel_prediction["win_prob"]:.1%}</span>', unsafe_allow_html=True)
                
                st.markdown(f"**Predicted Score:**")
                st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding: 0.5rem;'>{away_team} <strong>{excel_prediction['predicted_away']}</strong> - <strong>{excel_prediction['predicted_home']}</strong> {home_team}</div>", unsafe_allow_html=True)
                
                # HomeIce Differential with color
                st.markdown(f"**HomeIce Diff:**")
                st.markdown(f'<span class="diff-badge" style="color: {diff_color}; background: {diff_bg};">{excel_prediction["homeice_diff"]:+.3f}</span>', unsafe_allow_html=True)
        
        with col_ml:
            with st.container():
                st.markdown("#### 🤖 ML Model")
                if ml_prediction['has_ml']:
                    st.markdown(f'<div class="ml-winner">🏆 {ml_prediction["ml_predicted_winner"]}</div>', unsafe_allow_html=True)
                    ml_win_prob = ml_prediction['ml_home_win_prob'] if ml_prediction['ml_predicted_winner'] == home_team else ml_prediction['ml_away_win_prob']
                    
                    # ML Win Probability with color
                    ml_prob_color, ml_prob_bg = get_probability_color(ml_win_prob)
                    st.markdown(f"**Win Probability:**")
                    st.markdown(f'<span class="prob-badge" style="color: {ml_prob_color}; background: {ml_prob_bg};">{ml_win_prob:.1%}</span>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Predicted Score:**")
                    st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding: 0.5rem;'>{away_team} <strong>{ml_prediction['ml_predicted_away']}</strong> - <strong>{ml_prediction['ml_predicted_home']}</strong> {home_team}</div>", unsafe_allow_html=True)
                    
                    # ML Confidence with color
                    ml_conf_color, ml_conf_bg = get_probability_color(ml_prediction['ml_confidence'])
                    st.markdown(f"**Confidence:**")
                    st.markdown(f'<span class="diff-badge" style="color: {ml_conf_color}; background: {ml_conf_bg};">{ml_prediction["ml_confidence"]:.1%}</span>', unsafe_allow_html=True)
                else:
                    st.info("No ML prediction available")
        
        # Agreement indicator - centered
        if ml_prediction['has_ml']:
            st.markdown("<br>", unsafe_allow_html=True)
            if excel_prediction['predicted_winner'] == ml_prediction['ml_predicted_winner']:
                st.success("✅ Both models agree on winner")
            else:
                st.warning("⚠️ Models disagree on winner")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

def display_custom_matchup(home_team, away_team, standings, ml_predictions):
    """Display custom matchup with predictions - Modern Layout"""
    excel_prediction = calculate_prediction(home_team, away_team, standings)
    
    # Try to find ML prediction (might not exist for custom matchups)
    today = datetime.now()
    ml_prediction = get_ml_prediction(home_team, away_team, today, ml_predictions)
    
    home_row = excel_prediction['home_row']
    away_row = excel_prediction['away_row']
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Teams display - centered
        col1, col_vs, col2 = st.columns([2.5, 0.5, 2.5], gap="medium")
        
        with col1:
            st.markdown(f"<div class='team-display'>", unsafe_allow_html=True)
            st.markdown(f"### {away_team}")
            st.caption(f"**{int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])}** ({int(away_row['PTS'])} pts)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_vs:
            st.markdown("<div style='text-align: center; padding-top: 1.5rem; color: #666666;'><h2>@</h2></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div class='team-display'>", unsafe_allow_html=True)
            st.markdown(f"### {home_team}")
            st.caption(f"**{int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['OTL'])}** ({int(home_row['PTS'])} pts)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Get colors for probability and differential
        prob_color, prob_bg = get_probability_color(excel_prediction['win_prob'])
        diff_color, diff_bg = get_differential_color(excel_prediction['homeice_diff'])
        
        # Show predictions side by side
        col_excel, col_ml = st.columns(2, gap="large")
        
        with col_excel:
            with st.container():
                st.markdown("#### 📊 Excel Model")
                st.markdown(f'<div class="winner">🏆 {excel_prediction["predicted_winner"]}</div>', unsafe_allow_html=True)
                
                # Win Probability with color
                st.markdown(f"**Win Probability:**")
                st.markdown(f'<span class="prob-badge" style="color: {prob_color}; background: {prob_bg};">{excel_prediction["win_prob"]:.1%}</span>', unsafe_allow_html=True)
                
                st.markdown(f"**Predicted Score:**")
                st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding: 0.5rem;'>{away_team} <strong>{excel_prediction['predicted_away']}</strong> - <strong>{excel_prediction['predicted_home']}</strong> {home_team}</div>", unsafe_allow_html=True)
                
                # HomeIce Differential with color
                st.markdown(f"**HomeIce Diff:**")
                st.markdown(f'<span class="diff-badge" style="color: {diff_color}; background: {diff_bg};">{excel_prediction["homeice_diff"]:+.3f}</span>', unsafe_allow_html=True)
        
        with col_ml:
            with st.container():
                st.markdown("#### 🤖 ML Model")
                if ml_prediction['has_ml']:
                    st.markdown(f'<div class="ml-winner">🏆 {ml_prediction["ml_predicted_winner"]}</div>', unsafe_allow_html=True)
                    ml_win_prob = ml_prediction['ml_home_win_prob'] if ml_prediction['ml_predicted_winner'] == home_team else ml_prediction['ml_away_win_prob']
                    
                    # ML Win Probability with color
                    ml_prob_color, ml_prob_bg = get_probability_color(ml_win_prob)
                    st.markdown(f"**Win Probability:**")
                    st.markdown(f'<span class="prob-badge" style="color: {ml_prob_color}; background: {ml_prob_bg};">{ml_win_prob:.1%}</span>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Predicted Score:**")
                    st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding: 0.5rem;'>{away_team} <strong>{ml_prediction['ml_predicted_away']}</strong> - <strong>{ml_prediction['ml_predicted_home']}</strong> {home_team}</div>", unsafe_allow_html=True)
                    
                    # ML Confidence with color
                    ml_conf_color, ml_conf_bg = get_probability_color(ml_prediction['ml_confidence'])
                    st.markdown(f"**Confidence:**")
                    st.markdown(f'<span class="diff-badge" style="color: {ml_conf_color}; background: {ml_conf_bg};">{ml_prediction["ml_confidence"]:.1%}</span>', unsafe_allow_html=True)
                else:
                    st.info("No ML prediction available for this matchup")
        
        # Team stats comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Team Statistics")
        st.markdown("<br>", unsafe_allow_html=True)
        
        stats_data = {
            "Stat": ["Record", "Points", "Win %", "Goals Per Game", "Goals Allowed"],
            away_team: [
                f"{int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])}",
                int(away_row['PTS']),
                f"{away_row['AwayWin%']:.1%}",
                f"{away_row.iloc[15]:.2f}",
                f"{away_row.iloc[17]:.2f}"
            ],
            home_team: [
                f"{int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['OTL'])}",
                int(home_row['PTS']),
                f"{home_row['HomeWin%']:.1%}",
                f"{home_row.iloc[14]:.2f}",
                f"{home_row.iloc[16]:.2f}"
            ]
        }
        st.dataframe(stats_data, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

def main():
    # Load data
    predictions, standings, ml_predictions, load_timestamp = load_data()
    
    if predictions is None or standings is None:
        return
    
    # Title - Centered
    st.markdown("<div class='centered-container'>", unsafe_allow_html=True)
    st.title("🏒 NHL Prediction Model 2025-26")
    st.markdown(f"<div style='text-align: center; color: #666666; margin-bottom: 2rem;'>📅 {datetime.now().strftime('%A, %B %d, %Y')}</div>", unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Data", help="Clear cache and reload Excel file", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.caption(f"📊 Data loaded: {load_timestamp}")
    st.sidebar.markdown("---")
    
    # Determine available pages based on ML data
    has_ml_data = ml_predictions is not None and len(ml_predictions) > 0
    
    if has_ml_data:
        pages = ["Today's Games", "Custom Matchup", "Past Predictions", "Model Performance", "Model Comparison"]
    else:
        pages = ["Today's Games", "Custom Matchup", "Model Performance"]
    
    page = st.sidebar.radio("Select Page", pages)
    
    # TODAY'S GAMES PAGE
    if page == "Today's Games":
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Get today's games
        today = datetime.now().date()
        todays_games = predictions[predictions['Date'].dt.date == today].copy()
        
        if len(todays_games) == 0:
            st.warning("⚠️ No games scheduled for today")
            
            # Show upcoming games
            future_games = predictions[predictions['Date'].dt.date > today].copy()
            if len(future_games) > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📅 Next Upcoming Games")
                future_games = future_games.sort_values('Date').head(5)
                
                for idx, game in future_games.iterrows():
                    date_str = game['Date'].strftime('%A, %B %d')
                    st.info(f"**{game['Visitor']} @ {game['Home']}** - {date_str}")
        else:
            st.subheader(f"🏒 Today's Games ({len(todays_games)} matchups)")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Sort by time
            todays_games = todays_games.sort_values('Time')
            
            # Display each game - full width
            for idx, game in todays_games.iterrows():
                display_game_card(game, standings, ml_predictions)
    
    # CUSTOM MATCHUP PAGE
    elif page == "Custom Matchup":
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⚙️ Custom Matchup Prediction")
        st.markdown("<br>", unsafe_allow_html=True)
        
        teams = sorted(standings['Team'].tolist())
        
        # Team selectors
        col1, col_vs, col2 = st.columns([2.5, 0.5, 2.5], gap="medium")
        
        with col1:
            away_team = st.selectbox("**Away Team**", [""] + teams, key="away")
        
        with col_vs:
            st.markdown("<div style='text-align: center; padding-top: 2rem; color: #666666;'><h3>@</h3></div>", unsafe_allow_html=True)
        
        with col2:
            home_team = st.selectbox("**Home Team**", [""] + teams, key="home")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Center the button
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
                if not away_team or not home_team:
                    st.warning("⚠️ Please select both teams")
                elif away_team == home_team:
                    st.error("❌ Please select different teams")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    display_custom_matchup(home_team, away_team, standings, ml_predictions)
    
    # PAST PREDICTIONS PAGE (NEW!)
    elif page == "Past Predictions":
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📜 Past Predictions vs Actual Results")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if ml_predictions is None or len(ml_predictions) == 0:
            st.info("⚠️ No past predictions available. ML Prediction Model sheet is needed.")
        else:
            # Filter for games with actual results
            past_games = ml_predictions[ml_predictions['actual_winner'].notna()].copy()
            
            if len(past_games) == 0:
                st.info("No games with actual results yet.")
            else:
                # Sort by date descending (most recent first)
                past_games = past_games.sort_values('date', ascending=False)
                
                # Add filters
                st.markdown("### 🔍 Filters")
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                
                with col_filter1:
                    # Date range filter
                    date_filter = st.selectbox(
                        "Date Range",
                        ["All Time", "Last 7 Days", "Last 14 Days", "Last 30 Days"],
                        index=0
                    )
                
                with col_filter2:
                    # Team filter
                    all_teams = sorted(list(set(past_games['home_team'].tolist() + past_games['away_team'].tolist())))
                    team_filter = st.selectbox(
                        "Team",
                        ["All Teams"] + all_teams,
                        index=0
                    )
                
                with col_filter3:
                    # Result filter
                    result_filter = st.selectbox(
                        "Show",
                        ["All Games", "Both Correct", "Both Wrong", "Only Excel Correct", "Only ML Correct", "Models Disagreed"],
                        index=0
                    )
                
                # Apply filters
                filtered_games = past_games.copy()
                
                # Date filter
                if date_filter != "All Time":
                    days_map = {"Last 7 Days": 7, "Last 14 Days": 14, "Last 30 Days": 30}
                    cutoff_date = datetime.now() - pd.Timedelta(days=days_map[date_filter])
                    filtered_games = filtered_games[filtered_games['date'] >= cutoff_date]
                
                # Team filter
                if team_filter != "All Teams":
                    filtered_games = filtered_games[
                        (filtered_games['home_team'] == team_filter) | 
                        (filtered_games['away_team'] == team_filter)
                    ]
                
                # Result filter
                if result_filter == "Both Correct":
                    filtered_games = filtered_games[(filtered_games['ml_correct'] == "YES") & (filtered_games['excel_correct'] == "YES")]
                elif result_filter == "Both Wrong":
                    filtered_games = filtered_games[(filtered_games['ml_correct'] == "NO") & (filtered_games['excel_correct'] == "NO")]
                elif result_filter == "Only Excel Correct":
                    filtered_games = filtered_games[(filtered_games['excel_correct'] == "YES") & (filtered_games['ml_correct'] == "NO")]
                elif result_filter == "Only ML Correct":
                    filtered_games = filtered_games[(filtered_games['ml_correct'] == "YES") & (filtered_games['excel_correct'] == "NO")]
                elif result_filter == "Models Disagreed":
                    filtered_games = filtered_games[filtered_games['ml_predicted_winner'] != filtered_games['excel_predicted_winner']]
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Summary stats for filtered data
                if len(filtered_games) > 0:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"#### 📊 Summary ({len(filtered_games)} games)")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    excel_correct = (filtered_games['excel_correct'] == "YES").sum()
                    ml_correct = (filtered_games['ml_correct'] == "YES").sum()
                    both_correct = ((filtered_games['excel_correct'] == "YES") & (filtered_games['ml_correct'] == "YES")).sum()
                    agreement = (filtered_games['ml_predicted_winner'] == filtered_games['excel_predicted_winner']).sum()
                    
                    with col1:
                        st.metric("Excel Accuracy", f"{excel_correct/len(filtered_games)*100:.1f}%")
                    with col2:
                        st.metric("ML Accuracy", f"{ml_correct/len(filtered_games)*100:.1f}%")
                    with col3:
                        st.metric("Both Correct", f"{both_correct/len(filtered_games)*100:.1f}%")
                    with col4:
                        st.metric("Agreement Rate", f"{agreement/len(filtered_games)*100:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display games table
                    st.markdown("### 📋 Game Results")
                    
                    # Prepare display data
                    display_data = []
                    for idx, game in filtered_games.iterrows():
                        # Format predictions with correct/incorrect indicators
                        excel_indicator = "✅" if game['excel_correct'] == "YES" else "❌"
                        ml_indicator = "✅" if game['ml_correct'] == "YES" else "❌"
                        
                        # Agreement indicator
                        agree = "✅" if game['ml_predicted_winner'] == game['excel_predicted_winner'] else "⚠️"
                        
                        display_data.append({
                            "Date": game['date'].strftime('%Y-%m-%d'),
                            "Time": game['game_time'],
                            "Matchup": f"{game['away_team']} @ {game['home_team']}",
                            "Actual Winner": game['actual_winner'],
                            "Excel Prediction": f"{excel_indicator} {game['excel_predicted_winner']}",
                            "Excel Confidence": f"{game['excel_confidence']:.1%}",
                            "ML Prediction": f"{ml_indicator} {game['ml_predicted_winner']}",
                            "ML Confidence": f"{game['ml_confidence']:.1%}",
                            "Models Agree": agree
                        })
                    
                    # Display as dataframe
                    df_display = pd.DataFrame(display_data)
                    st.dataframe(df_display, use_container_width=True, hide_index=True, height=600)
                    
                else:
                    st.warning("No games match the selected filters.")
    
    # MODEL PERFORMANCE PAGE
    elif page == "Model Performance":
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Model Performance Statistics")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Excel Model Performance
        st.markdown("### 📊 Excel Model")
        correct_col = 'Locked Correct'
        completed = predictions[predictions[correct_col].isin(['YES', 'NO'])].copy()
        
        if len(completed) == 0:
            st.info("No completed games yet for Excel model")
        else:
            completed = completed.sort_values('Date')
            
            # Overall stats
            total_games = len(completed)
            correct_games = (completed[correct_col] == 'YES').sum()
            overall_accuracy = (correct_games / total_games * 100)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Overall Results")
            
            col1, col2, col3, col4 = st.columns(4, gap="medium")
            
            with col1:
                st.metric("Total Games", total_games)
            
            with col2:
                st.metric("Correct", correct_games)
            
            with col3:
                st.metric("Wrong", total_games - correct_games)
            
            with col4:
                st.metric("Accuracy", f"{overall_accuracy:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Last 20 games
            today = datetime.now()
            past_games = completed[completed['Date'] < today]
            
            if len(past_games) >= 20:
                st.markdown("<br>", unsafe_allow_html=True)
                last_20 = past_games.tail(20)
                correct_20 = (last_20[correct_col] == 'YES').sum()
                accuracy_20 = (correct_20 / 20 * 100)
                
                first_date = last_20['Date'].min().strftime('%Y-%m-%d')
                last_date = last_20['Date'].max().strftime('%Y-%m-%d')
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("#### 📊 Last 20 Games")
                st.caption(f"{first_date} to {last_date}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Correct", f"{correct_20}/20")
                
                with col2:
                    st.metric("Accuracy", f"{accuracy_20:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # ML Model Performance
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🤖 ML Model")
        
        if ml_predictions is None or len(ml_predictions) == 0:
            st.info("⚠️ ML Model predictions not available. Upload an Excel file with 'ML Prediction Model' sheet to see ML performance.")
        else:
            ml_completed = ml_predictions[ml_predictions['ml_correct'].notna()].copy()
            
            if len(ml_completed) == 0:
                st.info("No completed games yet for ML model")
            else:
                ml_completed = ml_completed.sort_values('date')
                
                # Overall stats
                ml_total = len(ml_completed)
                ml_correct = (ml_completed['ml_correct'] == "YES").sum()
                ml_accuracy = (ml_correct / ml_total * 100)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("#### 📈 Overall Results")
                
                col1, col2, col3, col4 = st.columns(4, gap="medium")
                
                with col1:
                    st.metric("Total Games", ml_total)
                
                with col2:
                    st.metric("Correct", ml_correct)
                
                with col3:
                    st.metric("Wrong", ml_total - ml_correct)
                
                with col4:
                    st.metric("Accuracy", f"{ml_accuracy:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Average confidence on correct vs incorrect
                if ml_total > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                    correct_conf = ml_completed[ml_completed['ml_correct'] == "YES"]['ml_confidence'].mean()
                    incorrect_conf = ml_completed[ml_completed['ml_correct'] == "NO"]['ml_confidence'].mean()
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("#### 🎯 Confidence Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Avg Confidence (Correct)", f"{correct_conf:.1%}")
                    
                    with col2:
                        st.metric("Avg Confidence (Incorrect)", f"{incorrect_conf:.1%}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # MODEL COMPARISON PAGE
    elif page == "Model Comparison":
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔄 Excel vs ML Model Comparison")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Get games where both models have predictions
        ml_completed = ml_predictions[ml_predictions['ml_correct'].notna()].copy()
        
        if len(ml_completed) == 0:
            st.info("No completed games to compare yet")
        else:
            ml_completed = ml_completed.sort_values('date')
            
            # Overall comparison
            ml_total = len(ml_completed)
            ml_correct = (ml_completed['ml_correct'] == "YES").sum()
            excel_correct = (ml_completed['excel_correct'] == "YES").sum()
            
            ml_accuracy = (ml_correct / ml_total * 100)
            excel_accuracy = (excel_correct / ml_total * 100)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Head-to-Head Comparison")
            st.caption(f"Based on {ml_total} completed games")
            
            col1, col2, col3 = st.columns(3, gap="large")
            
            with col1:
                st.metric("Excel Model Accuracy", f"{excel_accuracy:.1f}%")
                st.caption(f"{excel_correct}/{ml_total} correct")
            
            with col2:
                st.metric("ML Model Accuracy", f"{ml_accuracy:.1f}%")
                st.caption(f"{ml_correct}/{ml_total} correct")
            
            with col3:
                diff = ml_accuracy - excel_accuracy
                st.metric("Difference", f"{diff:+.1f}%")
                if diff > 0:
                    st.caption("🤖 ML Model leads")
                elif diff < 0:
                    st.caption("📊 Excel Model leads")
                else:
                    st.caption("🤝 Tied")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Agreement analysis
            st.markdown("<br>", unsafe_allow_html=True)
            both_correct = ((ml_completed['ml_correct'] == "YES") & (ml_completed['excel_correct'] == "YES")).sum()
            both_wrong = ((ml_completed['ml_correct'] == "NO") & (ml_completed['excel_correct'] == "NO")).sum()
            ml_only = ((ml_completed['ml_correct'] == "YES") & (ml_completed['excel_correct'] == "NO")).sum()
            excel_only = ((ml_completed['ml_correct'] == "NO") & (ml_completed['excel_correct'] == "YES")).sum()
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🤝 Model Agreement")
            
            col1, col2, col3, col4 = st.columns(4, gap="medium")
            
            with col1:
                st.metric("Both Correct", both_correct)
                st.caption(f"{both_correct/ml_total*100:.1f}%")
            
            with col2:
                st.metric("Both Wrong", both_wrong)
                st.caption(f"{both_wrong/ml_total*100:.1f}%")
            
            with col3:
                st.metric("ML Only Correct", ml_only)
                st.caption(f"{ml_only/ml_total*100:.1f}%")
            
            with col4:
                st.metric("Excel Only Correct", excel_only)
                st.caption(f"{excel_only/ml_total*100:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent games breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📋 Recent Games Breakdown")
            st.markdown("<br>", unsafe_allow_html=True)
            
            recent_games = ml_completed.tail(10)
            
            display_data = []
            for idx, game in recent_games.iterrows():
                excel_result = "✅" if game['excel_correct'] == "YES" else "❌"
                ml_result = "✅" if game['ml_correct'] == "YES" else "❌"
                
                display_data.append({
                    "Date": game['date'].strftime('%Y-%m-%d'),
                    "Matchup": f"{game['away_team']} @ {game['home_team']}",
                    "Winner": game['actual_winner'],
                    "Excel": f"{excel_result} {game['excel_predicted_winner']}",
                    "ML": f"{ml_result} {game['ml_predicted_winner']}"
                })
            
            st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    # Close centered container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    
    if ml_predictions is not None and len(ml_predictions) > 0:
        st.sidebar.info("""
        **NHL Prediction Model 2025-26**
        
        🔄 **Two Models:**
        - 📊 Excel: HomeIce Differential (70% team stats, 30% home ice)
        - 🤖 ML: Random Forest machine learning model
        
        Compare predictions and performance across both approaches!
        """)
    else:
        st.sidebar.info("""
        **NHL Prediction Model 2025-26**
        
        📊 **Excel Model**
        Using HomeIce Differential and team statistics to predict game outcomes.
        
        Model blends:
        - 70% team-based stats
        - 30% HomeIce advantage
        
        💡 Add 'ML Prediction Model' sheet to Excel file to enable ML predictions!
        """)

if __name__ == "__main__":
    main()
