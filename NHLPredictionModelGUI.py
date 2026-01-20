#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:21:07 2025

@author: aidanconte
"""

"""
NHL Prediction Model - Web App
Displays Excel and ML predictions side by side
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Page config
st.set_page_config(
    page_title="NHL Prediction Model 2025-26",
    page_icon="🏒",
    layout="wide"
)

# DraftKings/FanDuel Style CSS
st.markdown("""
    <style>
    /* Main styling - Dark theme like DraftKings/FanDuel */
    .main {
        background: #0d1117;
        padding: 1rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark background for entire app */
    .stApp {
        background: #0d1117;
    }
    
    /* Typography - White text on dark */
    h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Sidebar dark theme */
    .css-1d391kg {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Radio buttons styled */
    .stRadio > div {
        background: #161b22;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stRadio label {
        color: #c9d1d9;
        font-weight: 500;
    }
    
    /* Buttons - Green accent like DraftKings */
    .stButton > button {
        background: #00d4aa;
        color: #0d1117;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #00b894;
        transform: translateY(-1px);
    }
    
    /* Selectbox dark theme */
    .stSelectbox {
        margin-bottom: 1rem;
    }
    
    .stSelectbox label {
        color: #c9d1d9 !important;
        font-weight: 500;
    }
    
    .stSelectbox > div > div {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }
    
    .stSelectbox > div > div > select {
        background: #161b22 !important;
        color: #c9d1d9 !important;
    }
    
    /* Dataframe dark theme */
    .dataframe {
        background: #161b22;
        color: #c9d1d9;
    }
    
    /* Caption text */
    .stCaption {
        color: #8b949e;
    }
    
    /* Info boxes */
    .stInfo {
        background: #161b22;
        border-left: 4px solid #00d4aa;
        color: #c9d1d9;
    }
    
    .stSuccess {
        background: #161b22;
        border-left: 4px solid #00d4aa;
        color: #c9d1d9;
    }
    
    .stWarning {
        background: #161b22;
        border-left: 4px solid #ffa500;
        color: #c9d1d9;
    }
    
    .stError {
        background: #161b22;
        border-left: 4px solid #f85149;
        color: #c9d1d9;
    }
    
    /* Betting Card Style - Like DraftKings/FanDuel */
    .betting-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.2s ease;
    }
    
    .betting-card:hover {
        border-color: #00d4aa;
        box-shadow: 0 0 0 1px #00d4aa;
    }
    
    /* Game Header */
    .game-header {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Team Display */
    .team-display {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #30363d;
    }
    
    .team-name-betting {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .team-record-betting {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 0.25rem;
    }
    
    /* Odds Display - Like DraftKings */
    .odds-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        text-align: center;
        min-width: 100px;
    }
    
    .odds-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00d4aa;
    }
    
    .odds-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        margin-top: 0.25rem;
    }
    
    /* Score Prediction */
    .score-prediction {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        padding: 0.5rem 0;
    }
    
    /* Model Badge */
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-excel {
        background: #1f6feb;
        color: #ffffff;
    }
    
    .badge-ml {
        background: #a371f7;
        color: #ffffff;
    }
    
    /* Winner Highlight */
    .winner-highlight {
        color: #00d4aa;
        font-weight: 700;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-card {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d4aa;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 0.5rem;
    }
    
    /* Table Styling */
    table {
        background: #161b22;
        color: #c9d1d9;
    }
    
    thead {
        background: #0d1117;
        color: #ffffff;
    }
    
    tbody tr {
        border-bottom: 1px solid #30363d;
    }
    
    tbody tr:hover {
        background: #21262d;
    }
    
    </style>
""", unsafe_allow_html=True)

# Constants
EXCEL_FILE = 'Aidan Conte NHL 2025-26 Prediction Model.xlsx'
EASTERN = pytz.timezone('America/New_York')  # Eastern Time

@st.cache_data(ttl=3600)
def load_data():
    """Load Excel data"""
    try:
        # Load predictions
        excel_predictions_raw = pd.read_excel(EXCEL_FILE, sheet_name='NHL HomeIce Model', header=None)
        header_row = excel_predictions_raw.iloc[0].values
        predictions_data = excel_predictions_raw.iloc[1:].reset_index(drop=True)
        predictions_data.columns = header_row
        predictions = predictions_data.iloc[:, 3:].copy()
        
        # Normalize dates to remove timezone and time components
        predictions['Date'] = pd.to_datetime(predictions['Date']).dt.normalize()
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        # Load ML predictions
        ml_predictions = None
        try:
            ml_predictions = pd.read_excel(EXCEL_FILE, sheet_name='ML Prediction Model', header=0)
            
            if 'game_date' in ml_predictions.columns:
                # Normalize dates to remove timezone and time components
                ml_predictions['date'] = pd.to_datetime(ml_predictions['game_date'], errors='coerce').dt.normalize()
            
            # Rename columns
            rename_map = {
                'ml_winner': 'ml_predicted_winner',
                'ml_home_score': 'ml_predicted_home_score',
                'ml_away_score': 'ml_predicted_away_score',
                'ml_ot': 'ml_is_overtime',
                'ml_confidence': 'ml_confidence'
            }
            ml_predictions = ml_predictions.rename(columns={k: v for k, v in rename_map.items() if k in ml_predictions.columns})
            
            # Convert percentage strings
            if 'ml_confidence' in ml_predictions.columns:
                ml_predictions['ml_confidence'] = ml_predictions['ml_confidence'].apply(
                    lambda x: float(str(x).strip('%')) / 100.0 if isinstance(x, str) and '%' in str(x) else float(x) if pd.notna(x) else 0.5
                )
            
            # Convert OT to boolean
            if 'ml_is_overtime' in ml_predictions.columns:
                ml_predictions['ml_is_overtime'] = ml_predictions['ml_is_overtime'].apply(
                    lambda x: str(x).upper() in ['YES', 'TRUE', '1'] if pd.notna(x) else False
                )
            
        except Exception as e:
            st.sidebar.error(f"ML Model loading failed: {str(e)}")
            ml_predictions = pd.DataFrame()
        
        return predictions, standings, ml_predictions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data(ttl=3600)
def load_player_stats():
    """Load player stats from CSV"""
    try:
        player_stats_file = 'NHL2025-26PlayerStats.csv'
        # Try to read the CSV file
        df = pd.read_csv(player_stats_file, sep='~', encoding='utf-8', errors='ignore')
        # Clean up the data - remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        # Remove rows that are all empty or just headers
        df = df[df.iloc[:, 0].notna()]
        return df
    except Exception as e:
        st.sidebar.warning(f"Player stats file not found: {str(e)}")
        return None

def calculate_excel_prediction(home_team, away_team, standings, predictions, game_date):
    """Calculate Excel model prediction"""
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    # Get prediction from sheet if available
    predicted_winner = None
    homeice_diff = None
    
    if predictions is not None and game_date is not None:
        try:
            # Normalize both dates for comparison
            if isinstance(game_date, str):
                game_date_normalized = pd.to_datetime(game_date).normalize()
            elif hasattr(game_date, 'normalize'):
                game_date_normalized = game_date.normalize()
            else:
                game_date_normalized = pd.Timestamp(game_date).normalize()
            
            game_match = predictions[
                (predictions['Home'] == home_team) & 
                (predictions['Visitor'] == away_team) &
                (predictions['Date'] == game_date_normalized)
            ]
            
            if len(game_match) > 0:
                game = game_match.iloc[0]
                predicted_winner = game['Predicted Winner']
                homeice_diff = game['HomeIce Differential']
        except:
            pass
    
    # Calculate if not found
    if predicted_winner is None:
        home_home_win_pct = home_row['HomeWin%']
        away_away_win_pct = away_row['AwayWin%']
        homeice_diff = (home_home_win_pct - away_away_win_pct) * 6
        predicted_winner = home_team if homeice_diff > 0 else away_team
    
    # Calculate scores
    home_goals_for = home_row['Home Goals per Game']
    home_goals_against = home_row['Home Goals Against']
    away_goals_for = away_row['Away Goals per Game']
    away_goals_against = away_row['Away Goals Against']
    
    expected_home = (home_goals_for + away_goals_against) / 2
    expected_away = (away_goals_for + home_goals_against) / 2
    
    home_adjustment = homeice_diff * 0.5
    predicted_home = max(2, min(7, round(expected_home + home_adjustment)))
    predicted_away = max(2, min(7, round(expected_away - home_adjustment)))
    
    # Ensure winner has more goals
    if predicted_winner == home_team and predicted_home <= predicted_away:
        predicted_home = predicted_away + 1
    elif predicted_winner == away_team and predicted_away <= predicted_home:
        predicted_away = predicted_home + 1
    
    # Calculate probabilities
    win_prob = 0.5 + (abs(homeice_diff) / 12)
    win_prob = min(0.85, max(0.52, win_prob))
    
    # OT probability
    abs_diff = abs(homeice_diff)
    if abs_diff < 0.2:
        ot_prob = 0.40
    elif abs_diff < 0.5:
        ot_prob = 0.25
    elif abs_diff < 1.0:
        ot_prob = 0.15
    else:
        ot_prob = 0.08
    
    return {
        'winner': predicted_winner,
        'home_score': predicted_home,
        'away_score': predicted_away,
        'confidence': win_prob,
        'ot_probability': ot_prob,
        'homeice_diff': homeice_diff
    }

def get_ml_prediction(home_team, away_team, game_date, ml_predictions):
    """Get ML model prediction"""
    if ml_predictions is None or len(ml_predictions) == 0:
        return None
    
    # Normalize game_date to remove time component
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date).normalize()
    elif hasattr(game_date, 'normalize'):
        game_date = game_date.normalize()
    else:
        game_date = pd.Timestamp(game_date).normalize()
    
    try:
        # Compare normalized timestamps
        ml_game = ml_predictions[
            (ml_predictions['home_team'] == home_team) & 
            (ml_predictions['away_team'] == away_team) &
            (ml_predictions['date'] == game_date)
        ]
    except:
        return None
    
    if len(ml_game) == 0:
        return None
    
    ml_game = ml_game.iloc[0]
    
    # Get scores
    try:
        ml_home_score = int(float(ml_game.get('ml_predicted_home_score', 3)))
        ml_away_score = int(float(ml_game.get('ml_predicted_away_score', 2)))
    except:
        ml_home_score = 3
        ml_away_score = 2
    
    ml_is_overtime = ml_game.get('ml_is_overtime', False)
    ml_confidence = ml_game.get('ml_confidence', 0.5)
    
    return {
        'winner': ml_game['ml_predicted_winner'],
        'home_score': ml_home_score,
        'away_score': ml_away_score,
        'confidence': ml_confidence,
        'is_overtime': ml_is_overtime
    }

def probability_to_american_odds(prob):
    """Convert probability to American odds format"""
    if prob >= 0.5:
        odds = -((prob * 100) / (1 - prob))
        return f"{int(odds)}"
    else:
        odds = ((1 - prob) * 100) / prob
        return f"+{int(odds)}"

def display_game(game, standings, predictions, ml_predictions):
    """Display game in DraftKings/FanDuel betting card style"""
    home_team = game['Home']
    away_team = game['Visitor']
    game_time = game['Time']
    game_date = game['Date']
    
    # Get predictions
    excel_pred = calculate_excel_prediction(home_team, away_team, standings, predictions, game_date)
    ml_pred = get_ml_prediction(home_team, away_team, game_date, ml_predictions)
    
    # Get team records
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    st.markdown('<div class="betting-card">', unsafe_allow_html=True)
    
    # Game header
    st.markdown(f'<div class="game-header">{game_time} ET</div>', unsafe_allow_html=True)
    
    # Excel Model Prediction
    st.markdown('<span class="model-badge badge-excel">EXCEL MODEL</span>', unsafe_allow_html=True)
    
    # Teams and predictions
    col1, col2, col3 = st.columns([3, 1, 2])
    
    with col1:
        st.markdown(f'<div class="team-name-betting">{away_team}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="team-record-betting">{int(away_row["W"])}-{int(away_row["L"])}-{int(away_row["OTL"])} | {int(away_row["PTS"])} PTS</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="score-prediction">VS</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="team-name-betting" style="text-align: right;">{home_team}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="team-record-betting" style="text-align: right;">{int(home_row["W"])}-{int(home_row["L"])}-{int(home_row["OTL"])} | {int(home_row["PTS"])} PTS</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
    
    # Prediction details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="odds-box">', unsafe_allow_html=True)
        winner_class = "winner-highlight" if excel_pred['winner'] == away_team else ""
        st.markdown(f'<div class="odds-value {winner_class}">{excel_pred["away_score"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="odds-label">Away Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="odds-box">', unsafe_allow_html=True)
        home_win_prob = excel_pred['confidence'] if excel_pred['winner'] == home_team else (1 - excel_pred['confidence'])
        away_win_prob = excel_pred['confidence'] if excel_pred['winner'] == away_team else (1 - excel_pred['confidence'])
        excel_away_odds = probability_to_american_odds(away_win_prob)
        excel_home_odds = probability_to_american_odds(home_win_prob)
        st.markdown(f'<div class="odds-value">{excel_away_odds}</div>', unsafe_allow_html=True)
        st.markdown('<div class="odds-label">Odds</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="odds-box">', unsafe_allow_html=True)
        winner_class = "winner-highlight" if excel_pred['winner'] == home_team else ""
        st.markdown(f'<div class="odds-value {winner_class}">{excel_pred["home_score"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="odds-label">Home Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="text-align: center; margin-top: 1rem; color: #8b949e; font-size: 0.85rem;">Predicted Winner: <span class="winner-highlight">{excel_pred["winner"]}</span> | Confidence: {excel_pred["confidence"]:.1%}</div>', unsafe_allow_html=True)
    
    # ML Model if available
    if ml_pred:
        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
        st.markdown('<span class="model-badge badge-ml">ML MODEL</span>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="odds-box">', unsafe_allow_html=True)
            winner_class = "winner-highlight" if ml_pred['winner'] == away_team else ""
            st.markdown(f'<div class="odds-value {winner_class}">{ml_pred["away_score"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="odds-label">Away Score</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="odds-box">', unsafe_allow_html=True)
            ml_away_win_prob = ml_pred['confidence'] if ml_pred['winner'] == away_team else ml_pred.get('away_win_probability', 1 - ml_pred['confidence'])
            ml_away_odds = probability_to_american_odds(ml_away_win_prob)
            st.markdown(f'<div class="odds-value">{ml_away_odds}</div>', unsafe_allow_html=True)
            st.markdown('<div class="odds-label">Odds</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="odds-box">', unsafe_allow_html=True)
            winner_class = "winner-highlight" if ml_pred['winner'] == home_team else ""
            st.markdown(f'<div class="odds-value {winner_class}">{ml_pred["home_score"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="odds-label">Home Score</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        ot_text = " | Overtime Predicted" if ml_pred.get('is_overtime', False) else ""
        st.markdown(f'<div style="text-align: center; margin-top: 1rem; color: #8b949e; font-size: 0.85rem;">Predicted Winner: <span class="winner-highlight">{ml_pred["winner"]}</span> | Confidence: {ml_pred["confidence"]:.1%}{ot_text}</div>', unsafe_allow_html=True)
        
        # Agreement
        if excel_pred['winner'] == ml_pred['winner']:
            st.markdown('<div style="text-align: center; margin-top: 1rem; padding: 0.5rem; background: #0d1117; border-radius: 4px; color: #00d4aa; font-weight: 600;">✓ Both Models Agree</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; margin-top: 1rem; padding: 0.5rem; background: #0d1117; border-radius: 4px; color: #ffa500; font-weight: 600;">⚠ Models Disagree</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    predictions, standings, ml_predictions = load_data()
    
    if predictions is None or standings is None:
        st.error("❌ Error loading data. Please check your Excel file.")
        return
    
    # DraftKings/FanDuel Style Header
    st.markdown("""
        <div style='background: #161b22; border-bottom: 2px solid #00d4aa; padding: 1.5rem 0; margin-bottom: 2rem;'>
            <div style='max-width: 1200px; margin: 0 auto; padding: 0 1rem;'>
                <h1 style='color: #ffffff; margin: 0; font-size: 2rem; font-weight: 700;'>NHL PREDICTIONS</h1>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>2025-26 Season • Advanced Analytics & Machine Learning</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    eastern_now = datetime.now(EASTERN)
    st.caption(f"Last updated: {eastern_now.strftime('%Y-%m-%d %I:%M %p')} ET")
    
    # Sidebar Navigation
    st.sidebar.markdown("""
        <div style='background: #161b22; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #30363d;'>
            <h3 style='color: #ffffff; margin: 0; font-size: 1rem;'>NAVIGATION</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Dropdown menu for navigation
    st.sidebar.markdown('<p style="color: #c9d1d9; font-weight: 600; margin-bottom: 0.75rem; font-size: 0.9rem;">SELECT PAGE</p>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        label="Choose a page",
        options=["Standings", "Player Leaderboard", "Model Performance"],
        index=0,
        label_visibility="visible",
        key="page_selector"
    )
    
    if page == "Standings":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>STANDINGS</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>2025-26 Season Team Rankings</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display standings table
        if standings is not None and len(standings) > 0:
            # Select key columns to display
            display_cols = ['Team', 'GP', 'W', 'L', 'OTL', 'PTS', 'HomeWin%', 'AwayWin%']
            available_cols = [col for col in display_cols if col in standings.columns]
            
            if len(available_cols) > 0:
                standings_display = standings[available_cols].copy()
                standings_display = standings_display.sort_values('PTS', ascending=False)
                
                # Format percentages
                if 'HomeWin%' in standings_display.columns:
                    standings_display['HomeWin%'] = standings_display['HomeWin%'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                if 'AwayWin%' in standings_display.columns:
                    standings_display['AwayWin%'] = standings_display['AwayWin%'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                
                # Rename columns for better display
                standings_display.columns = standings_display.columns.str.replace('HomeWin%', 'Home Win %')
                standings_display.columns = standings_display.columns.str.replace('AwayWin%', 'Away Win %')
                
                st.dataframe(
                    standings_display,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
            else:
                st.dataframe(standings, use_container_width=True, hide_index=True, height=600)
        else:
            st.error("❌ Unable to load standings data")
    
    elif page == "Player Leaderboard":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>PLAYER LEADERBOARD</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Top Player Statistics & Performance</p>
            </div>
        """, unsafe_allow_html=True)
        
        player_stats = load_player_stats()
        
        if player_stats is not None and len(player_stats) > 0:
            # Clean up column names
            player_stats.columns = player_stats.columns.str.strip()
            
            # Try to identify key columns
            key_cols = []
            possible_cols = {
                'Player': ['Player', 'player', 'Name', 'name'],
                'Team': ['Team', 'team', 'Tm', 'tm'],
                'GP': ['GP', 'gp', 'Games', 'games'],
                'G': ['G', 'g', 'Goals', 'goals'],
                'A': ['A', 'a', 'Assists', 'assists'],
                'PTS': ['PTS', 'pts', 'Points', 'points', 'P', 'p']
            }
            
            for display_name, possible_names in possible_cols.items():
                for col in player_stats.columns:
                    if col in possible_names:
                        key_cols.append(col)
                        break
            
            # If we found key columns, display them
            if len(key_cols) > 0:
                # Remove duplicates while preserving order
                key_cols = list(dict.fromkeys(key_cols))
                player_display = player_stats[key_cols].copy()
                
                # Try to sort by points if available
                pts_col = None
                for col in ['PTS', 'pts', 'Points', 'points', 'P', 'p']:
                    if col in player_display.columns:
                        pts_col = col
                        break
                
                if pts_col:
                    # Convert to numeric, handling any non-numeric values
                    player_display[pts_col] = pd.to_numeric(player_display[pts_col], errors='coerce')
                    player_display = player_display.sort_values(pts_col, ascending=False, na_position='last')
                    player_display = player_display.head(100)  # Show top 100
                
                st.dataframe(
                    player_display,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
            else:
                # Display all columns if we can't identify key ones
                st.dataframe(
                    player_stats.head(100),
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
            
            st.caption(f"📊 Showing data for {len(player_stats)} players")
        else:
            st.warning("⚠️ Player stats data not available. Please ensure NHL2025-26PlayerStats.csv exists in the project directory.")
    
    elif page == "Model Performance":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>MODEL PERFORMANCE & COMPARISON</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Track accuracy and compare model predictions</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Excel performance with DraftKings styling
        st.markdown("### Excel Model Performance")
        completed = predictions[predictions['Locked Correct'].isin(['YES', 'NO'])].copy()
        if len(completed) > 0:
            total = len(completed)
            correct = (completed['Locked Correct'] == 'YES').sum()
            wrong = total - correct
            accuracy = (correct / total * 100)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{total}</div>
                        <div class="stat-label">Total Games</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{correct}</div>
                        <div class="stat-label">Correct</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value" style="color: #f85149;">{wrong}</div>
                        <div class="stat-label">Incorrect</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{accuracy:.1f}%</div>
                        <div class="stat-label">Accuracy</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Progress bar for accuracy
            st.markdown(f"""
                <div style='margin-top: 1.5rem; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 1rem;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span style='font-weight: 600; color: #c9d1d9;'>Overall Accuracy</span>
                        <span style='font-weight: 600; color: #00d4aa;'>{accuracy:.1f}%</span>
                    </div>
                    <div style='background: #21262d; border-radius: 6px; height: 24px; overflow: hidden; border: 1px solid #30363d;'>
                        <div style='background: #00d4aa; 
                                    height: 100%; width: {accuracy}%; display: flex; align-items: center; 
                                    justify-content: center; color: #0d1117; font-weight: 600; font-size: 0.85rem;'>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("📊 No completed games yet")
        
        st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
        
        # ML performance with DraftKings styling
        st.markdown("### ML Model Performance")
        if ml_predictions is not None and len(ml_predictions) > 0:
            ml_completed = ml_predictions[pd.notna(ml_predictions.get('ml_correct', pd.Series()))].copy()
            if len(ml_completed) > 0:
                # Convert YES/NO to 1/0
                ml_completed['ml_correct_num'] = ml_completed['ml_correct'].apply(
                    lambda x: 1 if str(x).upper() == 'YES' else 0 if str(x).upper() == 'NO' else np.nan
                )
                ml_completed = ml_completed[pd.notna(ml_completed['ml_correct_num'])]
                
                if len(ml_completed) > 0:
                    ml_total = len(ml_completed)
                    ml_correct = ml_completed['ml_correct_num'].sum()
                    ml_wrong = ml_total - ml_correct
                    ml_accuracy = (ml_correct / ml_total * 100)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value">{ml_total}</div>
                                <div class="stat-label">Total Games</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value">{int(ml_correct)}</div>
                                <div class="stat-label">Correct</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value" style="color: #f85149;">{int(ml_wrong)}</div>
                                <div class="stat-label">Incorrect</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value" style="color: #a371f7;">{ml_accuracy:.1f}%</div>
                                <div class="stat-label">Accuracy</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Progress bar for ML accuracy
                    st.markdown(f"""
                        <div style='margin-top: 1.5rem; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 1rem;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                                <span style='font-weight: 600; color: #c9d1d9;'>ML Model Accuracy</span>
                                <span style='font-weight: 600; color: #a371f7;'>{ml_accuracy:.1f}%</span>
                            </div>
                            <div style='background: #21262d; border-radius: 6px; height: 24px; overflow: hidden; border: 1px solid #30363d;'>
                                <div style='background: #a371f7; 
                                            height: 100%; width: {ml_accuracy}%; display: flex; align-items: center; 
                                            justify-content: center; color: #ffffff; font-weight: 600; font-size: 0.85rem;'>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("📊 No completed games yet")
            else:
                st.info("📊 No completed games yet")
        else:
            st.info("🤖 ML model not available")

if __name__ == "__main__":
    main()
