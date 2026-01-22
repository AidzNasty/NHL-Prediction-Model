#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHL Prediction Model - Web App with Top Scorers
Displays Excel and ML predictions side by side + probable scorers
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
    
    /* Betting Card Style */
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
    
    /* Scorer Card Style */
    .scorer-card {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.35rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .scorer-name {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.9rem;
        display: block;
    }
    
    .scorer-team {
        color: #8b949e;
        font-size: 0.75rem;
        display: block;
        margin-top: 0.1rem;
    }
    
    .scorer-stat {
        color: #00d4aa;
        font-weight: 700;
        font-size: 0.95rem;
        white-space: nowrap;
        margin-left: 0.5rem;
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
    
    </style>
""", unsafe_allow_html=True)

# Constants
EXCEL_FILE = 'Aidan Conte NHL 2025-26 Prediction Model.xlsx'
EASTERN = pytz.timezone('America/New_York')

@st.cache_data(ttl=3600)
def load_data():
    """Load Excel data"""
    try:
        # Load Excel predictions from NHL HomeIce Model sheet
        excel_raw = pd.read_excel(EXCEL_FILE, sheet_name='NHL HomeIce Model', header=None)
        
        # Extract header from row 0, starting at column 3 (index 3)
        header_row = excel_raw.iloc[0, 3:].values
        
        # Extract data starting from row 1, columns 3 onwards
        predictions_data = excel_raw.iloc[1:, 3:].copy()
        predictions_data.columns = header_row
        
        # Remove NaN columns
        valid_cols = [col for col in predictions_data.columns if pd.notna(col)]
        predictions = predictions_data[valid_cols].copy()
        
        # Normalize dates to timezone-naive and filter out invalid dates
        predictions['Date'] = pd.to_datetime(predictions['Date'], errors='coerce')
        # Remove timezone if present
        if hasattr(predictions['Date'].dtype, 'tz') and predictions['Date'].dtype.tz is not None:
            predictions['Date'] = predictions['Date'].dt.tz_localize(None)
        # Normalize to date only (no time component)
        predictions['Date'] = predictions['Date'].dt.normalize()
        
        # Filter out rows with NaT (Not a Time) dates to avoid comparison errors
        predictions = predictions[pd.notna(predictions['Date'])].copy()
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        # Load ML predictions
        ml_predictions = None
        try:
            ml_predictions = pd.read_excel(EXCEL_FILE, sheet_name='ML Prediction Model', header=0)
            
            if 'game_date' in ml_predictions.columns:
                ml_predictions['date'] = pd.to_datetime(ml_predictions['game_date'], errors='coerce')
                # Remove timezone if present
                if hasattr(ml_predictions['date'].dtype, 'tz') and ml_predictions['date'].dtype.tz is not None:
                    ml_predictions['date'] = ml_predictions['date'].dt.tz_localize(None)
                ml_predictions['date'] = ml_predictions['date'].dt.normalize()
        except Exception as e:
            st.sidebar.warning(f"ML Model loading failed: {str(e)}")
            ml_predictions = pd.DataFrame()
        
        return predictions, standings, ml_predictions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data(ttl=3600)
def load_player_stats():
    """Load player stats from Excel and calculate per-game averages"""
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name='NHL2025-26PlayerStats')
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Calculate per-game stats
        if 'GP' in df.columns:
            if 'G' in df.columns:
                df['Goals_Per_Game'] = df['G'] / df['GP'].replace(0, 1)
            if 'A' in df.columns:
                df['Assists_Per_Game'] = df['A'] / df['GP'].replace(0, 1)
            if 'PTS' in df.columns:
                df['Points_Per_Game'] = df['PTS'] / df['GP'].replace(0, 1)
        
        return df
    except Exception as e:
        st.sidebar.warning(f"Error loading player stats: {str(e)}")
        return None

def calculate_player_probabilities(home_team, away_team, standings, player_stats, excel_pred):
    """Calculate scoring probabilities for players based on game matchup"""
    if player_stats is None or standings is None or len(player_stats) == 0:
        return None
    
    try:
        # Get team stats
        home_stats = standings[standings['Team'] == home_team].iloc[0]
        away_stats = standings[standings['Team'] == away_team].iloc[0]
        
        # Get players from both teams
        home_players = player_stats[player_stats['Team'] == home_team].copy()
        away_players = player_stats[player_stats['Team'] == away_team].copy()
        
        if len(home_players) == 0 and len(away_players) == 0:
            return None
        
        # Expected team goals from predictions
        if excel_pred:
            expected_home_goals = excel_pred['home_score']
            expected_away_goals = excel_pred['away_score']
        else:
            expected_home_goals = home_stats['Home Goals per Game']
            expected_away_goals = away_stats['Away Goals per Game']
        
        # Calculate goal probability for each player
        # Probability = (Player GPG / Team GPG) * Expected Team Goals
        home_team_gpg = home_stats.get('Home Goals per Game', 3.0)
        away_team_gpg = away_stats.get('Away Goals per Game', 3.0)
        
        # Avoid division by zero
        if home_team_gpg == 0:
            home_team_gpg = 3.0
        if away_team_gpg == 0:
            away_team_gpg = 3.0
        
        player_dfs = []
        
        if len(home_players) > 0 and 'Goals_Per_Game' in home_players.columns:
            home_players['Goal_Probability'] = (home_players['Goals_Per_Game'] / home_team_gpg) * expected_home_goals
            home_players['Assist_Probability'] = (home_players['Assists_Per_Game'] / (home_team_gpg * 2)) * expected_home_goals * 2
            home_players['Point_Probability'] = home_players['Goal_Probability'] + home_players['Assist_Probability']
            player_dfs.append(home_players)
        
        if len(away_players) > 0 and 'Goals_Per_Game' in away_players.columns:
            away_players['Goal_Probability'] = (away_players['Goals_Per_Game'] / away_team_gpg) * expected_away_goals
            away_players['Assist_Probability'] = (away_players['Assists_Per_Game'] / (away_team_gpg * 2)) * expected_away_goals * 2
            away_players['Point_Probability'] = away_players['Goal_Probability'] + away_players['Assist_Probability']
            player_dfs.append(away_players)
        
        if len(player_dfs) == 0:
            return None
        
        # Combine players from both teams
        all_players = pd.concat(player_dfs, ignore_index=True)
        
        # Get top scorers
        top_goals = all_players.nlargest(5, 'Goal_Probability')[['Player', 'Team', 'Pos', 'Goals_Per_Game', 'Goal_Probability']]
        top_assists = all_players.nlargest(5, 'Assist_Probability')[['Player', 'Team', 'Pos', 'Assists_Per_Game', 'Assist_Probability']]
        top_points = all_players.nlargest(5, 'Point_Probability')[['Player', 'Team', 'Pos', 'Points_Per_Game', 'Point_Probability']]
        
        return {
            'goals': top_goals,
            'assists': top_assists,
            'points': top_points
        }
    except Exception as e:
        st.sidebar.error(f"Error calculating probabilities: {e}")
        return None

def calculate_excel_prediction(home_team, away_team, standings, predictions, game_date):
    """Calculate Excel model prediction using HomeIce Differential"""
    try:
        home_row = standings[standings['Team'] == home_team].iloc[0]
        away_row = standings[standings['Team'] == away_team].iloc[0]
    except:
        return None
    
    # Get prediction from sheet if available
    predicted_winner = None
    homeice_diff = None
    
    if predictions is not None and game_date is not None:
        try:
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
    
    # Calculate confidence from HomeIce Differential
    abs_diff = abs(homeice_diff)
    confidence = 0.5 + (abs_diff / 12)
    confidence = min(0.85, max(0.52, confidence))
    
    # OT probability based on HomeIce Differential
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
        'confidence': confidence,
        'ot_probability': ot_prob,
        'homeice_diff': homeice_diff
    }

def get_ml_prediction(home_team, away_team, game_date, ml_predictions):
    """Get ML model prediction from Excel sheet"""
    if ml_predictions is None or len(ml_predictions) == 0:
        return None
    
    # Normalize game_date
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date).normalize()
    elif hasattr(game_date, 'normalize'):
        game_date = game_date.normalize()
    else:
        game_date = pd.Timestamp(game_date).normalize()
    
    try:
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
        ml_home_score = int(float(ml_game.get('ml_home_score', 3)))
        ml_away_score = int(float(ml_game.get('ml_away_score', 2)))
    except:
        ml_home_score = 3
        ml_away_score = 2
    
    # Get OT prediction
    ml_ot = ml_game.get('ml_ot', 'NO')
    if isinstance(ml_ot, str):
        ml_is_overtime = ml_ot.upper() in ['YES', 'TRUE', '1']
    else:
        ml_is_overtime = bool(ml_ot)
    
    # Get confidence
    ml_confidence = ml_game.get('ml_confidence', 0.5)
    if isinstance(ml_confidence, str) and '%' in str(ml_confidence):
        ml_confidence = float(str(ml_confidence).strip('%')) / 100.0
    else:
        ml_confidence = float(ml_confidence) if pd.notna(ml_confidence) else 0.5
    
    # Calculate OT probability (if ML predicts OT, set high probability)
    ot_probability = 0.75 if ml_is_overtime else 0.15
    
    return {
        'winner': ml_game['ml_winner'],
        'home_score': ml_home_score,
        'away_score': ml_away_score,
        'confidence': ml_confidence,
        'ot_probability': ot_probability,
        'is_overtime': ml_is_overtime
    }

def display_daily_pick(game_row, standings, predictions, ml_predictions, player_stats=None, is_excel=True):
    """Display a single game's pick with top scorers"""
    if is_excel:
        home_team = game_row['Home']
        away_team = game_row['Visitor']
        game_time = game_row['Time']
        game_date = game_row['Date']
        
        excel_pred = calculate_excel_prediction(home_team, away_team, standings, predictions, game_date)
        ml_pred = get_ml_prediction(home_team, away_team, game_date, ml_predictions)
    else:
        # ML row
        home_team = game_row['home_team']
        away_team = game_row['away_team']
        game_time = game_row['game_time']
        game_date = game_row['date']
        
        excel_pred = calculate_excel_prediction(home_team, away_team, standings, predictions, game_date)
        ml_pred = get_ml_prediction(home_team, away_team, game_date, ml_predictions)
    
    if excel_pred is None and ml_pred is None:
        return
    
    # Get team records
    try:
        home_row = standings[standings['Team'] == home_team].iloc[0]
        away_row = standings[standings['Team'] == away_team].iloc[0]
    except:
        return
    
    st.markdown('<div class="betting-card">', unsafe_allow_html=True)
    
    # Game header
    time_str = str(game_time) if pd.notna(game_time) else "TBD"
    if ':' not in time_str:
        time_str = "TBD"
    st.markdown(f'<div style="color: #8b949e; font-size: 0.85rem; font-weight: 500; margin-bottom: 1rem; text-transform: uppercase;">{time_str} ET</div>', unsafe_allow_html=True)
    
    # Teams
    col1, col2, col3 = st.columns([3, 1, 2])
    
    with col1:
        st.markdown(f'<div style="font-size: 1.2rem; font-weight: 600; color: #ffffff;">{away_team}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.25rem;">{int(away_row["W"])}-{int(away_row["L"])}-{int(away_row["OTL"])} | {int(away_row["PTS"])} PTS</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #ffffff; text-align: center; padding: 0.5rem 0;">@</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div style="font-size: 1.2rem; font-weight: 600; color: #ffffff; text-align: right;">{home_team}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.25rem; text-align: right;">{int(home_row["W"])}-{int(home_row["L"])}-{int(home_row["OTL"])} | {int(home_row["PTS"])} PTS</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
    
    # Excel Prediction
    if excel_pred:
        st.markdown('<span class="model-badge badge-excel">EXCEL MODEL</span>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Pick</div>
                    <div class='winner-highlight' style='font-size: 1.1rem; margin-top: 0.25rem;'>{excel_pred['winner']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Score</div>
                    <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{excel_pred['away_score']}-{excel_pred['home_score']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>OT Prob</div>
                    <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{excel_pred['ot_probability']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Confidence</div>
                    <div style='color: #00d4aa; font-size: 1.1rem; margin-top: 0.25rem;'>{excel_pred['confidence']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # ML Prediction
    if ml_pred:
        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
        st.markdown('<span class="model-badge badge-ml">ML MODEL</span>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Pick</div>
                    <div class='winner-highlight' style='font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['winner']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Score</div>
                    <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['away_score']}-{ml_pred['home_score']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>OT Prob</div>
                    <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['ot_probability']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Confidence</div>
                    <div style='color: #a371f7; font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['confidence']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Agreement indicator
        if excel_pred and excel_pred['winner'] == ml_pred['winner']:
            st.markdown('<div style="text-align: center; margin-top: 1rem; padding: 0.5rem; background: #0d1117; border-radius: 4px; color: #00d4aa; font-weight: 600;">✓ Both Models Agree</div>', unsafe_allow_html=True)
        elif excel_pred:
            st.markdown('<div style="text-align: center; margin-top: 1rem; padding: 0.5rem; background: #0d1117; border-radius: 4px; color: #ffa500; font-weight: 600;">⚠ Models Disagree</div>', unsafe_allow_html=True)
    
    # TOP PROBABLE SCORERS FOR THIS GAME
    if player_stats is not None:
        scorer_data = calculate_player_probabilities(home_team, away_team, standings, player_stats, excel_pred)
        
        if scorer_data:
            st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
            st.markdown('<h3 style="color: #ffffff; font-size: 1.1rem; margin-bottom: 1rem; text-align: center;">🎯 TOP PROBABLE SCORERS</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            # Goals
            with col1:
                st.markdown('<div style="color: #8b949e; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase;">⚽ Goals</div>', unsafe_allow_html=True)
                if len(scorer_data['goals']) > 0:
                    for idx, player in scorer_data['goals'].iterrows():
                        prob = player['Goal_Probability'] * 100
                        gpg = player['Goals_Per_Game']
                        st.markdown(f"""
                            <div class="scorer-card">
                                <div style="flex: 1;">
                                    <div class="scorer-name">{player['Player']}</div>
                                    <div class="scorer-team">{player['Team']} • {player['Pos']}</div>
                                </div>
                                <div class="scorer-stat">{prob:.1f}%</div>
                            </div>
                            <div style="color: #8b949e; font-size: 0.7rem; margin: -0.25rem 0 0.5rem 0.75rem;">{gpg:.2f} goals/game</div>
                        """, unsafe_allow_html=True)
            
            # Assists
            with col2:
                st.markdown('<div style="color: #8b949e; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase;">🎯 Assists</div>', unsafe_allow_html=True)
                if len(scorer_data['assists']) > 0:
                    for idx, player in scorer_data['assists'].iterrows():
                        prob = player['Assist_Probability'] * 100
                        apg = player['Assists_Per_Game']
                        st.markdown(f"""
                            <div class="scorer-card">
                                <div style="flex: 1;">
                                    <div class="scorer-name">{player['Player']}</div>
                                    <div class="scorer-team">{player['Team']} • {player['Pos']}</div>
                                </div>
                                <div class="scorer-stat">{prob:.1f}%</div>
                            </div>
                            <div style="color: #8b949e; font-size: 0.7rem; margin: -0.25rem 0 0.5rem 0.75rem;">{apg:.2f} assists/game</div>
                        """, unsafe_allow_html=True)
            
            # Points
            with col3:
                st.markdown('<div style="color: #8b949e; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase;">🏆 Points</div>', unsafe_allow_html=True)
                if len(scorer_data['points']) > 0:
                    for idx, player in scorer_data['points'].iterrows():
                        prob = player['Point_Probability'] * 100
                        ppg = player['Points_Per_Game']
                        st.markdown(f"""
                            <div class="scorer-card">
                                <div style="flex: 1;">
                                    <div class="scorer-name">{player['Player']}</div>
                                    <div class="scorer-team">{player['Team']} • {player['Pos']}</div>
                                </div>
                                <div class="scorer-stat">{prob:.1f}%</div>
                            </div>
                            <div style="color: #8b949e; font-size: 0.7rem; margin: -0.25rem 0 0.5rem 0.75rem;">{ppg:.2f} points/game</div>
                        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    """Display a single game's pick"""
    if is_excel:
        home_team = game_row['Home']
        away_team = game_row['Visitor']
        game_time = game_row['Time']
        game_date = game_row['Date']
        
        excel_pred = calculate_excel_prediction(home_team, away_team, standings, predictions, game_date)
        ml_pred = get_ml_prediction(home_team, away_team, game_date, ml_predictions)
    else:
        # ML row
        home_team = game_row['home_team']
        away_team = game_row['away_team']
        game_time = game_row['game_time']
        game_date = game_row['date']
        
        excel_pred = calculate_excel_prediction(home_team, away_team, standings, predictions, game_date)
        ml_pred = get_ml_prediction(home_team, away_team, game_date, ml_predictions)
    
    if excel_pred is None and ml_pred is None:
        return
    
    # Get team records
    try:
        home_row = standings[standings['Team'] == home_team].iloc[0]
        away_row = standings[standings['Team'] == away_team].iloc[0]
    except:
        return
    
    st.markdown('<div class="betting-card">', unsafe_allow_html=True)
    
    # Game header
    time_str = str(game_time) if pd.notna(game_time) else "TBD"
    if ':' not in time_str:
        time_str = "TBD"
    st.markdown(f'<div style="color: #8b949e; font-size: 0.85rem; font-weight: 500; margin-bottom: 1rem; text-transform: uppercase;">{time_str} ET</div>', unsafe_allow_html=True)
    
    # Teams
    col1, col2, col3 = st.columns([3, 1, 2])
    
    with col1:
        st.markdown(f'<div style="font-size: 1.2rem; font-weight: 600; color: #ffffff;">{away_team}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.25rem;">{int(away_row["W"])}-{int(away_row["L"])}-{int(away_row["OTL"])} | {int(away_row["PTS"])} PTS</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #ffffff; text-align: center; padding: 0.5rem 0;">@</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div style="font-size: 1.2rem; font-weight: 600; color: #ffffff; text-align: right;">{home_team}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.25rem; text-align: right;">{int(home_row["W"])}-{int(home_row["L"])}-{int(home_row["OTL"])} | {int(home_row["PTS"])} PTS</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
    
    # Excel Prediction
    if excel_pred:
        st.markdown('<span class="model-badge badge-excel">EXCEL MODEL</span>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Pick</div>
                    <div class='winner-highlight' style='font-size: 1.1rem; margin-top: 0.25rem;'>{excel_pred['winner']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Score</div>
                    <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{excel_pred['away_score']}-{excel_pred['home_score']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>OT Prob</div>
                    <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{excel_pred['ot_probability']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Confidence</div>
                    <div style='color: #00d4aa; font-size: 1.1rem; margin-top: 0.25rem;'>{excel_pred['confidence']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # ML Prediction
    if ml_pred:
        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
        st.markdown('<span class="model-badge badge-ml">ML MODEL</span>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Pick</div>
                    <div class='winner-highlight' style='font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['winner']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Score</div>
                    <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['away_score']}-{ml_pred['home_score']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>OT Prob</div>
                    <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['ot_probability']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Confidence</div>
                    <div style='color: #a371f7; font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['confidence']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Agreement indicator
        if excel_pred and excel_pred['winner'] == ml_pred['winner']:
            st.markdown('<div style="text-align: center; margin-top: 1rem; padding: 0.5rem; background: #0d1117; border-radius: 4px; color: #00d4aa; font-weight: 600;">✓ Both Models Agree</div>', unsafe_allow_html=True)
        elif excel_pred:
            st.markdown('<div style="text-align: center; margin-top: 1rem; padding: 0.5rem; background: #0d1117; border-radius: 4px; color: #ffa500; font-weight: 600;">⚠ Models Disagree</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    predictions, standings, ml_predictions = load_data()
    
    if predictions is None or standings is None:
        st.error("❌ Error loading data. Please check your Excel file.")
        return
    
    # Header
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
    
    # Top Dropdown Menu
    st.markdown("""
        <div style='background: #161b22; border: 1px solid #30363d; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;'>
            <p style='color: #c9d1d9; font-weight: 600; margin: 0 0 0.5rem 0; font-size: 0.9rem;'>📊 SELECT PAGE</p>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.selectbox(
        label="Navigate to:",
        options=["🏒 Daily Picks", "📊 Standings", "🏆 Player Leaderboard", "📈 Model Performance"],
        index=0,
        label_visibility="collapsed",
        key="page_selector"
    )
    
    # Sidebar
    st.sidebar.markdown("""
        <div style='background: #161b22; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #30363d;'>
            <h3 style='color: #ffffff; margin: 0; font-size: 1rem;'>QUICK ACTIONS</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # PAGE CONTENT
    if page == "🏒 Daily Picks":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>DAILY PICKS</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Today's predictions from both models with top probable scorers</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Load player stats
        player_stats = load_player_stats()
        
        # Get today's games - use timezone-naive date for comparison
        today = pd.Timestamp.now().normalize()
        
        if 'Date' in predictions.columns:
            # Get today's games
            todays_games = predictions[predictions['Date'] == today].copy()
            
            if len(todays_games) > 0:
                st.success(f"🏒 {len(todays_games)} games scheduled for today")
                
                # Display game picks with scorers
                for idx, game in todays_games.iterrows():
                    display_daily_pick(game, standings, predictions, ml_predictions, player_stats, is_excel=True)
            else:
                st.info("📅 No games scheduled for today")
                
                # Show upcoming games
                upcoming = predictions[predictions['Date'] > today].sort_values('Date').head(5)
                if len(upcoming) > 0:
                    st.markdown("### 📆 Next Upcoming Games")
                    for idx, game in upcoming.iterrows():
                        display_daily_pick(game, standings, predictions, ml_predictions, player_stats, is_excel=True)
        else:
            st.error("❌ Date column not found in predictions")
    
    elif page == "📊 Standings":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>STANDINGS</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>2025-26 Season Team Rankings</p>
            </div>
        """, unsafe_allow_html=True)
        
        if standings is not None and len(standings) > 0:
            display_cols = ['Team', 'GP', 'W', 'L', 'OTL', 'PTS', 'HomeWin%', 'AwayWin%']
            available_cols = [col for col in display_cols if col in standings.columns]
            
            if len(available_cols) > 0:
                standings_display = standings[available_cols].copy()
                standings_display = standings_display.sort_values('PTS', ascending=False)
                
                if 'HomeWin%' in standings_display.columns:
                    standings_display['HomeWin%'] = standings_display['HomeWin%'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                if 'AwayWin%' in standings_display.columns:
                    standings_display['AwayWin%'] = standings_display['AwayWin%'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                
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
    
    elif page == "🏆 Player Leaderboard":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>PLAYER LEADERBOARD</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Top Player Statistics & Performance</p>
            </div>
        """, unsafe_allow_html=True)
        
        player_stats = load_player_stats()
        
        if player_stats is not None and len(player_stats) > 0:
            # Display key stats
            display_cols = ['Rk', 'Player', 'Age', 'Team', 'Pos', 'GP', 'G', 'A', 'PTS', '+/-', 'PIM', 'SOG', 'SPCT']
            available_cols = [col for col in display_cols if col in player_stats.columns]
            
            if len(available_cols) > 0:
                player_display = player_stats[available_cols].copy()
                
                # Sort by points
                if 'PTS' in player_display.columns:
                    player_display = player_display.sort_values('PTS', ascending=False)
                
                st.dataframe(
                    player_display.head(100),
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
            else:
                st.dataframe(
                    player_stats.head(100),
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
            
            st.caption(f"📊 Showing top 100 of {len(player_stats)} players")
        else:
            st.warning("⚠️ Player stats not available. Please ensure the Excel file contains a 'NHL2025-26PlayerStats' sheet.")
    
    elif page == "📈 Model Performance":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>MODEL PERFORMANCE</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Track accuracy and compare predictions</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Excel Model Performance
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
        else:
            st.info("📊 No completed games yet")
        
        st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
        
        # ML Model Performance
        st.markdown("### ML Model Performance")
        if ml_predictions is not None and len(ml_predictions) > 0:
            ml_completed = ml_predictions[pd.notna(ml_predictions.get('ml_correct', pd.Series()))].copy()
            if len(ml_completed) > 0:
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
                else:
                    st.info("📊 No completed games yet")
            else:
                st.info("📊 No completed games yet")
        else:
            st.info("🤖 ML model not available")

if __name__ == "__main__":
    main()
