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

# Enhanced Modern CSS with NHL Theme
st.markdown("""
    <style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        padding: 2rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    h1 {
        color: #1a1a2e;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #1a1a2e;
        font-weight: 600;
    }
    
    /* Game Card - Modern with shadow and gradient */
    .game-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 16px;
        border: none;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12), 0 2px 6px rgba(0, 0, 0, 0.08);
        margin: 1.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .game-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15), 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Prediction Box - Enhanced with NHL colors */
    .prediction-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        border-color: #003e7e;
        box-shadow: 0 6px 16px rgba(0, 62, 126, 0.15);
    }
    
    /* Excel Model Box - Blue theme */
    .excel-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-color: #2196f3;
    }
    
    /* ML Model Box - Purple/Teal theme */
    .ml-box {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-color: #9c27b0;
    }
    
    /* Winner Text - Bold and colorful */
    .winner-text {
        font-size: 1.4rem;
        font-weight: 700;
        color: #003e7e;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Score Text - Large and prominent */
    .score-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0.75rem 0;
        letter-spacing: 1px;
    }
    
    /* Metric Label - Subtle but readable */
    .metric-label {
        font-size: 0.95rem;
        color: #555;
        margin: 0.25rem 0;
        font-weight: 500;
    }
    
    /* Team Name Styling */
    .team-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #003e7e;
        margin-bottom: 0.5rem;
    }
    
    .team-record {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Agreement Indicator */
    .agreement-indicator {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        margin-top: 1rem;
        text-align: center;
    }
    
    .agreement-yes {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
    }
    
    .agreement-no {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
    }
    
    /* Game Time Header */
    .game-time {
        font-size: 1rem;
        color: #666;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        text-align: center;
        border-top: 4px solid #003e7e;
    }
    
    /* Custom Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #003e7e 0%, #0056b3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 62, 126, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 62, 126, 0.4);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
    }
    
    /* Radio Button Styling */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Info/Warning/Success Messages */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    /* Performance Metrics */
    .performance-metric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
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

def display_game(game, standings, predictions, ml_predictions):
    """Display a single game prediction with enhanced visuals"""
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
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Header with game time
        st.markdown(f'<div class="game-time">🕐 {game_time}</div>', unsafe_allow_html=True)
        
        # Teams with enhanced styling
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f'<div class="team-name">✈️ {away_team}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-record">📊 {int(away_row["W"])}-{int(away_row["L"])}-{int(away_row["OTL"])} | {int(away_row["PTS"])} pts</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("<h2 style='text-align: center; color: #003e7e; margin: 0.5rem 0;'>@</h2>", unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="team-name" style="text-align: right;">🏠 {home_team}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-record" style="text-align: right;">📊 {int(home_row["W"])}-{int(home_row["L"])}-{int(home_row["OTL"])} | {int(home_row["PTS"])} pts</div>', unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 2px solid #e0e0e0;'>", unsafe_allow_html=True)
        
        # Predictions with enhanced boxes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-box excel-box">', unsafe_allow_html=True)
            st.markdown("### 📊 Excel Model", unsafe_allow_html=True)
            st.markdown(f'<div class="winner-text">🏆 {excel_pred["winner"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-text">{away_team} <strong>{excel_pred["away_score"]}</strong> - <strong>{excel_pred["home_score"]}</strong> {home_team}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-badge">Confidence: {excel_pred["confidence"]:.1%}</div>', unsafe_allow_html=True)
            if excel_pred['ot_probability'] > 0.25:
                st.markdown(f'<div class="metric-label" style="margin-top: 0.5rem;">⏱️ OT Probability: <strong>{excel_pred["ot_probability"]:.1%}</strong></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-box ml-box">', unsafe_allow_html=True)
            st.markdown("### 🤖 ML Model", unsafe_allow_html=True)
            if ml_pred:
                st.markdown(f'<div class="winner-text">🏆 {ml_pred["winner"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-text">{away_team} <strong>{ml_pred["away_score"]}</strong> - <strong>{ml_pred["home_score"]}</strong> {home_team}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-badge">Confidence: {ml_pred["confidence"]:.1%}</div>', unsafe_allow_html=True)
                if ml_pred['is_overtime']:
                    st.markdown('<div class="metric-label" style="margin-top: 0.5rem; color: #9c27b0; font-weight: 600;">⏱️ Overtime Predicted</div>', unsafe_allow_html=True)
            else:
                st.info("🤖 No ML prediction available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Agreement indicator with enhanced styling
        if ml_pred:
            if excel_pred['winner'] == ml_pred['winner']:
                st.markdown('<div class="agreement-indicator agreement-yes">✅ Both models agree on winner: <strong>' + excel_pred['winner'] + '</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agreement-indicator agreement-no">⚠️ Models disagree: Excel → {excel_pred["winner"]} | ML → {ml_pred["winner"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    predictions, standings, ml_predictions = load_data()
    
    if predictions is None or standings is None:
        st.error("❌ Error loading data. Please check your Excel file.")
        return
    
    # Enhanced Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #003e7e 0%, #0056b3 100%); 
                    border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 8px 24px rgba(0, 62, 126, 0.3);'>
            <h1 style='color: white; margin: 0; padding: 0.5rem 0;'>🏒 NHL Prediction Model 2025-26</h1>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0; font-size: 1.1rem;'>Advanced Analytics & Machine Learning Predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    eastern_now = datetime.now(EASTERN)
    st.caption(f"🕐 Last updated: {eastern_now.strftime('%Y-%m-%d %I:%M:%S %p')} ET")
    
    # Enhanced Sidebar
    st.sidebar.markdown("""
        <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
            <h2 style='color: white; margin: 0; text-align: center;'>🧭 Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("📄 Select Page", ["Today's Games", "Custom Matchup", "Performance"], label_visibility="visible")
    
    if page == "Today's Games":
        # Get current date in Eastern Time
        eastern_now = datetime.now(EASTERN)
        today = pd.Timestamp(eastern_now.date()).normalize()
        
        todays_games = predictions[predictions['Date'] == today].copy()
        
        if len(todays_games) == 0:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                            padding: 2rem; border-radius: 12px; text-align: center; 
                            border-left: 4px solid #ff9800; margin: 2rem 0;'>
                    <h2 style='color: #856404; margin: 0 0 1rem 0;'>📅 No Games Scheduled Today</h2>
                    <p style='color: #856404; font-size: 1.1rem;'>Check out upcoming games below!</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show upcoming games with enhanced styling
            future_games = predictions[predictions['Date'] > today].head(5)
            if len(future_games) > 0:
                st.markdown("### 🔮 Upcoming Games")
                for _, game in future_games.iterrows():
                    st.markdown(f"""
                        <div style='background: white; padding: 1rem 1.5rem; border-radius: 8px; 
                                    margin: 0.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                                    border-left: 4px solid #003e7e;'>
                            <strong>{game['Visitor']}</strong> @ <strong>{game['Home']}</strong> 
                            <span style='color: #666; float: right;'>{game['Date'].strftime('%B %d, %Y')}</span>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                            padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;
                            border-left: 4px solid #4caf50;'>
                    <h2 style='color: #2e7d32; margin: 0;'>🏒 Today's Games ({len(todays_games)})</h2>
                    <p style='color: #2e7d32; margin: 0.5rem 0 0 0;'>{today.strftime('%A, %B %d, %Y')}</p>
                </div>
            """, unsafe_allow_html=True)
            todays_games = todays_games.sort_values('Time')
            for _, game in todays_games.iterrows():
                display_game(game, standings, predictions, ml_predictions)
    
    elif page == "Custom Matchup":
        st.markdown("""
            <div style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;
                        border-left: 4px solid #2196f3;'>
                <h2 style='color: #1565c0; margin: 0;'>🎯 Custom Matchup Predictor</h2>
                <p style='color: #1565c0; margin: 0.5rem 0 0 0;'>Select any two teams to see predictions</p>
            </div>
        """, unsafe_allow_html=True)
        
        teams = sorted(standings['Team'].tolist())
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✈️ Away Team")
            away_team = st.selectbox("", ["Select..."] + teams, label_visibility="collapsed", key="away_select")
        with col2:
            st.markdown("### 🏠 Home Team")
            home_team = st.selectbox("", ["Select..."] + teams, label_visibility="collapsed", key="home_select")
        
        if st.button("🚀 Generate Prediction", use_container_width=True, type="primary"):
            if away_team != "Select..." and home_team != "Select..." and away_team != home_team:
                eastern_now = datetime.now(EASTERN)
                today = pd.Timestamp(eastern_now.date()).normalize()
                excel_pred = calculate_excel_prediction(home_team, away_team, standings, None, None)
                ml_pred = get_ml_prediction(home_team, away_team, today, ml_predictions)
                
                # Enhanced prediction display
                st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
                st.markdown(f"### 🏒 {away_team} @ {home_team}", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="prediction-box excel-box">', unsafe_allow_html=True)
                    st.markdown("### 📊 Excel Model", unsafe_allow_html=True)
                    st.markdown(f'<div class="winner-text">🏆 {excel_pred["winner"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="score-text">{away_team} <strong>{excel_pred["away_score"]}</strong> - <strong>{excel_pred["home_score"]}</strong> {home_team}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-badge">Confidence: {excel_pred["confidence"]:.1%}</div>', unsafe_allow_html=True)
                    if excel_pred['ot_probability'] > 0.25:
                        st.markdown(f'<div class="metric-label" style="margin-top: 0.5rem;">⏱️ OT Probability: <strong>{excel_pred["ot_probability"]:.1%}</strong></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="prediction-box ml-box">', unsafe_allow_html=True)
                    st.markdown("### 🤖 ML Model", unsafe_allow_html=True)
                    if ml_pred:
                        st.markdown(f'<div class="winner-text">🏆 {ml_pred["winner"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="score-text">{away_team} <strong>{ml_pred["away_score"]}</strong> - <strong>{ml_pred["home_score"]}</strong> {home_team}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="confidence-badge">Confidence: {ml_pred["confidence"]:.1%}</div>', unsafe_allow_html=True)
                        if ml_pred['is_overtime']:
                            st.markdown('<div class="metric-label" style="margin-top: 0.5rem; color: #9c27b0; font-weight: 600;">⏱️ Overtime Predicted</div>', unsafe_allow_html=True)
                    else:
                        st.info("🤖 No ML prediction available")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Agreement indicator
                if ml_pred:
                    if excel_pred['winner'] == ml_pred['winner']:
                        st.markdown('<div class="agreement-indicator agreement-yes">✅ Both models agree on winner: <strong>' + excel_pred['winner'] + '</strong></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="agreement-indicator agreement-no">⚠️ Models disagree: Excel → {excel_pred["winner"]} | ML → {ml_pred["winner"]}</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Please select two different teams")
    
    elif page == "Performance":
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                        padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;
                        border-left: 4px solid #9c27b0;'>
                <h2 style='color: #7b1fa2; margin: 0;'>📈 Model Performance Analytics</h2>
                <p style='color: #7b1fa2; margin: 0.5rem 0 0 0;'>Track accuracy and performance metrics</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Excel performance with enhanced styling
        st.markdown("### 📊 Excel Model Performance")
        completed = predictions[predictions['Locked Correct'].isin(['YES', 'NO'])].copy()
        if len(completed) > 0:
            total = len(completed)
            correct = (completed['Locked Correct'] == 'YES').sum()
            wrong = total - correct
            accuracy = (correct / total * 100)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="metric-card" style='border-top-color: #2196f3;'>
                        <div style='font-size: 2rem; font-weight: 700; color: #2196f3;'>{total}</div>
                        <div style='color: #666; margin-top: 0.5rem;'>Total Games</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card" style='border-top-color: #4caf50;'>
                        <div style='font-size: 2rem; font-weight: 700; color: #4caf50;'>{correct}</div>
                        <div style='color: #666; margin-top: 0.5rem;'>Correct</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="metric-card" style='border-top-color: #f44336;'>
                        <div style='font-size: 2rem; font-weight: 700; color: #f44336;'>{wrong}</div>
                        <div style='color: #666; margin-top: 0.5rem;'>Incorrect</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class="metric-card" style='border-top-color: #ff9800;'>
                        <div style='font-size: 2rem; font-weight: 700; color: #ff9800;'>{accuracy:.1f}%</div>
                        <div style='color: #666; margin-top: 0.5rem;'>Accuracy</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Progress bar for accuracy
            st.markdown(f"""
                <div style='margin-top: 1.5rem;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span style='font-weight: 600; color: #333;'>Overall Accuracy</span>
                        <span style='font-weight: 600; color: #333;'>{accuracy:.1f}%</span>
                    </div>
                    <div style='background: #e0e0e0; border-radius: 10px; height: 30px; overflow: hidden;'>
                        <div style='background: linear-gradient(90deg, #4caf50 0%, #45a049 100%); 
                                    height: 100%; width: {accuracy}%; display: flex; align-items: center; 
                                    justify-content: center; color: white; font-weight: 600;'>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("📊 No completed games yet")
        
        st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 2px solid #e0e0e0;'>", unsafe_allow_html=True)
        
        # ML performance with enhanced styling
        st.markdown("### 🤖 ML Model Performance")
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
                            <div class="metric-card" style='border-top-color: #9c27b0;'>
                                <div style='font-size: 2rem; font-weight: 700; color: #9c27b0;'>{ml_total}</div>
                                <div style='color: #666; margin-top: 0.5rem;'>Total Games</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class="metric-card" style='border-top-color: #4caf50;'>
                                <div style='font-size: 2rem; font-weight: 700; color: #4caf50;'>{int(ml_correct)}</div>
                                <div style='color: #666; margin-top: 0.5rem;'>Correct</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                            <div class="metric-card" style='border-top-color: #f44336;'>
                                <div style='font-size: 2rem; font-weight: 700; color: #f44336;'>{int(ml_wrong)}</div>
                                <div style='color: #666; margin-top: 0.5rem;'>Incorrect</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                            <div class="metric-card" style='border-top-color: #ff9800;'>
                                <div style='font-size: 2rem; font-weight: 700; color: #ff9800;'>{ml_accuracy:.1f}%</div>
                                <div style='color: #666; margin-top: 0.5rem;'>Accuracy</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Progress bar for ML accuracy
                    st.markdown(f"""
                        <div style='margin-top: 1.5rem;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                                <span style='font-weight: 600; color: #333;'>ML Model Accuracy</span>
                                <span style='font-weight: 600; color: #333;'>{ml_accuracy:.1f}%</span>
                            </div>
                            <div style='background: #e0e0e0; border-radius: 10px; height: 30px; overflow: hidden;'>
                                <div style='background: linear-gradient(90deg, #9c27b0 0%, #7b1fa2 100%); 
                                            height: 100%; width: {ml_accuracy}%; display: flex; align-items: center; 
                                            justify-content: center; color: white; font-weight: 600;'>
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