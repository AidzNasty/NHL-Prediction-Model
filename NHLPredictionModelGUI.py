#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHL Prediction Model - Enhanced Web App with Excel Score Predictions
Includes: Score predictions, Overtime predictions based on HomeIce Differential
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="NHL Prediction Model 2025-26",
    page_icon="🏒",
    layout="wide"
)

# Custom CSS - Modern, Neutral Design
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
        color: #000000 !important;
        text-align: center;
    }
    h1 {
        margin-bottom: 0.5rem;
    }
    
    /* Centered Container - Wider for Desktop */
    .centered-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 2rem;
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
            max-width: 100%;
            padding: 0 1rem;
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
        color: #000000;
    }
    
    /* All text elements black */
    body, p, span, div, label, caption, .stText, .stMarkdown {
        color: #000000 !important;
    }
    
    /* Captions and labels */
    .stCaption, label {
        color: #000000 !important;
    }
    
    /* Better table styling */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
EXCEL_FILE = 'Aidan Conte NHL 2025-26 Prediction Model.xlsx'
# Alternative file names to try
EXCEL_FILE_ALTERNATIVES = [
    'Aidan Conte NHL 2025-26 Prediction Model.xlsx',
    'Aidan_Conte_NHL_2025-26_Prediction_Model.xlsx',
]

def find_excel_file():
    """Find the Excel file from alternatives"""
    import os
    for filename in EXCEL_FILE_ALTERNATIVES:
        if os.path.exists(filename):
            return filename
    return EXCEL_FILE_ALTERNATIVES[0]  # Return first as default

EXCEL_FILE = find_excel_file()
LEAGUE_AVG_TOTAL = 6.24

def get_probability_color(probability):
    """
    Convert probability (0-1) to color from red (weak) to green (strong)
    Returns hex color and background color
    """
    # Normalize probability to 0-1 range (assuming 0.5-1.0 is the range)
    # For win probability, 0.5 is weakest, 1.0 is strongest
    normalized = (probability - 0.5) / 0.5  # Maps 0.5->0, 1.0->1
    normalized = max(0, min(1, normalized))  # Clamp to 0-1
    
    # Red to Green gradient
    # Red: #d32f2f, Yellow: #fbc02d, Green: #388e3c
    if normalized < 0.5:
        # Red to Yellow
        t = normalized * 2
        r = int(211 + (251 - 211) * t)  # 211 -> 251
        g = int(47 + (192 - 47) * t)    # 47 -> 192
        b = int(47 + (45 - 47) * t)     # 47 -> 45
    else:
        # Yellow to Green
        t = (normalized - 0.5) * 2
        r = int(251 + (56 - 251) * t)   # 251 -> 56
        g = int(192 + (142 - 192) * t)  # 192 -> 142
        b = int(45 + (60 - 45) * t)     # 45 -> 60
    
    color = f"#{r:02x}{g:02x}{b:02x}"
    bg_color = f"rgba({r}, {g}, {b}, 0.15)"
    return color, bg_color

def get_differential_color(differential):
    """
    Convert homeice differential to color from red (negative/weak) to green (positive/strong)
    Returns hex color and background color
    """
    # Normalize differential to 0-1 range
    # Assuming range is roughly -3 to +3, map to 0-1
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

def convert_percentage_string(pct_str):
    """Convert percentage string like '96.00%' to float like 0.96"""
    try:
        if pd.isna(pct_str) or pct_str == 'nan%':
            return 0.0
        if isinstance(pct_str, str):
            return float(pct_str.strip('%')) / 100.0
        return float(pct_str)
    except:
        return 0.0

@st.cache_data(ttl=3600)  # Cache expires after 1 hour
def load_data():
    """Load Excel data with caching"""
    load_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Show which file we're using
    import os
    st.sidebar.caption(f"📁 Using file: {EXCEL_FILE}")
    st.sidebar.caption(f"📁 File exists: {os.path.exists(EXCEL_FILE)}")
    
    try:
        # Load NHL HomeIce Model sheet (predictions)
        excel_predictions_raw = pd.read_excel(EXCEL_FILE, sheet_name='NHL HomeIce Model', header=None)
        
        # Extract header row (row 0) and data starts from row 1
        header_row = excel_predictions_raw.iloc[0].values
        predictions_data = excel_predictions_raw.iloc[1:].reset_index(drop=True)
        predictions_data.columns = header_row
        
        # Select relevant columns starting from column 3 (Date onwards)
        predictions = predictions_data.iloc[:, 3:].copy()
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        # Try to load ML predictions (optional)
        ml_predictions = None
        ml_error_details = None
        try:
            st.sidebar.info("🔄 Loading ML Model...")
            ml_predictions = pd.read_excel(EXCEL_FILE, sheet_name='ML Prediction Model', header=0)
            st.sidebar.info(f"📊 Read {len(ml_predictions)} ML predictions")
            
            # Show actual columns for debugging
            st.sidebar.caption(f"Columns found: {', '.join(ml_predictions.columns[:5])}...")
            
            # Find date column - try multiple possible names
            date_col = None
            possible_date_cols = ['game_date', 'date', 'Date', 'GAME_DATE', 'Game Date']
            for col in possible_date_cols:
                if col in ml_predictions.columns:
                    date_col = col
                    break
            
            if date_col is None:
                raise ValueError(f"No date column found. Available columns: {list(ml_predictions.columns)}")
            
            # Convert date column
            ml_predictions['date'] = pd.to_datetime(ml_predictions[date_col])
            
            # Rename columns to match expected names - only rename if they exist
            rename_map = {}
            if 'ml_winner' in ml_predictions.columns:
                rename_map['ml_winner'] = 'ml_predicted_winner'
            if 'ml_home_score' in ml_predictions.columns:
                rename_map['ml_home_score'] = 'ml_predicted_home_score'
            if 'ml_away_score' in ml_predictions.columns:
                rename_map['ml_away_score'] = 'ml_predicted_away_score'
            if 'ml_ot' in ml_predictions.columns:
                rename_map['ml_ot'] = 'ml_is_overtime'
            if 'ml_ot_prob' in ml_predictions.columns:
                rename_map['ml_ot_prob'] = 'ml_overtime_probability'
            if 'excel_winner' in ml_predictions.columns:
                rename_map['excel_winner'] = 'excel_predicted_winner'
            
            if rename_map:
                ml_predictions = ml_predictions.rename(columns=rename_map)
                st.sidebar.caption(f"Renamed {len(rename_map)} columns")
            
            # Convert percentage strings to floats - only if columns exist
            if 'ml_confidence' in ml_predictions.columns:
                ml_predictions['ml_confidence'] = ml_predictions['ml_confidence'].apply(convert_percentage_string)
            else:
                ml_predictions['ml_confidence'] = 0.5  # Default confidence
                
            if 'ml_overtime_probability' in ml_predictions.columns:
                ml_predictions['ml_overtime_probability'] = ml_predictions['ml_overtime_probability'].apply(convert_percentage_string)
            else:
                ml_predictions['ml_overtime_probability'] = 0.0
            
            # Convert YES/NO to boolean for overtime
            if 'ml_is_overtime' in ml_predictions.columns:
                ml_predictions['ml_is_overtime'] = ml_predictions['ml_is_overtime'].apply(lambda x: x == 'YES' if pd.notna(x) else False)
            else:
                ml_predictions['ml_is_overtime'] = False
            
            # Calculate win probabilities (use confidence as base)
            # Check if necessary columns exist
            if all(col in ml_predictions.columns for col in ['ml_predicted_winner', 'home_team', 'ml_confidence']):
                ml_predictions['ml_home_win_prob'] = ml_predictions.apply(
                    lambda row: row['ml_confidence'] if row['ml_predicted_winner'] == row['home_team'] else (1 - row['ml_confidence']),
                    axis=1
                )
                ml_predictions['ml_away_win_prob'] = ml_predictions.apply(
                    lambda row: row['ml_confidence'] if row['ml_predicted_winner'] == row['away_team'] else (1 - row['ml_confidence']),
                    axis=1
                )
            else:
                ml_predictions['ml_home_win_prob'] = 0.5
                ml_predictions['ml_away_win_prob'] = 0.5
            
            # Convert correct columns: YES -> 1, NO -> 0, NaN -> NaN
            if 'ml_correct' in ml_predictions.columns:
                ml_predictions['ml_correct'] = ml_predictions['ml_correct'].apply(
                    lambda x: 1 if x == 'YES' else (0 if x == 'NO' else np.nan)
                )
            else:
                ml_predictions['ml_correct'] = np.nan
                
            if 'excel_correct' in ml_predictions.columns:
                ml_predictions['excel_correct'] = ml_predictions['excel_correct'].apply(
                    lambda x: 1 if x == 'YES' else (0 if x == 'NO' else np.nan)
                )
            else:
                ml_predictions['excel_correct'] = np.nan
            
            st.sidebar.success(f"✅ ML Model loaded ({len(ml_predictions)} games)")
            
        except Exception as ml_error:
            import traceback
            ml_error_details = traceback.format_exc()
            st.sidebar.error("❌ ML Model failed to load")
            st.sidebar.caption(f"Error: {str(ml_error)}")
            with st.sidebar.expander("Show full error"):
                st.code(ml_error_details)
            ml_predictions = pd.DataFrame()  # Empty dataframe
        
        return predictions, standings, ml_predictions, load_timestamp
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {e}")
        st.info("Make sure 'Aidan_Conte_NHL_2025-26_Prediction_Model.xlsx' is in the same folder as this script")
        return None, None, None, None

def calculate_overtime_probability(homeice_diff):
    """
    Calculate overtime probability based on HomeIce Differential
    Close games (small differential) are more likely to go to OT
    """
    # Use absolute value - closer to 0 means higher OT chance
    abs_diff = abs(homeice_diff)
    
    # Map differential to OT probability
    # 0.0 differential = ~45% OT chance
    # 0.5 differential = ~25% OT chance  
    # 1.0+ differential = ~10% OT chance
    
    if abs_diff < 0.2:
        ot_prob = 0.40 + (0.2 - abs_diff) * 0.5  # 40-50%
    elif abs_diff < 0.5:
        ot_prob = 0.25 + (0.5 - abs_diff) * 0.5  # 25-40%
    elif abs_diff < 1.0:
        ot_prob = 0.10 + (1.0 - abs_diff) * 0.3  # 10-25%
    else:
        ot_prob = max(0.05, 0.10 - (abs_diff - 1.0) * 0.05)  # 5-10%
    
    return max(0.05, min(0.50, ot_prob))  # Clamp between 5% and 50%

def predict_score_from_excel(home_team, away_team, predicted_winner, homeice_diff, standings):
    """
    Predict game score based on Excel's predicted winner and HomeIce Differential
    Uses team goals for/against to create realistic scores
    """
    # Get team stats
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    # Get goals per game stats
    home_goals_for = home_row['Home Goals per Game']
    home_goals_against = home_row['Home Goals Against']
    away_goals_for = away_row['Away Goals per Game']
    away_goals_against = away_row['Away Goals Against']
    
    # Calculate expected goals for each team (baseline)
    expected_home = (home_goals_for + away_goals_against) / 2
    expected_away = (away_goals_for + home_goals_against) / 2
    
    # Apply HomeIce Differential adjustment
    # Positive differential favors home team, negative favors away
    home_adjustment = homeice_diff * 0.5
    away_adjustment = -homeice_diff * 0.5
    
    adjusted_home = expected_home + home_adjustment
    adjusted_away = expected_away + away_adjustment
    
    # Ensure the predicted winner actually wins
    if predicted_winner == home_team:
        # Home team should win
        predicted_home = round(adjusted_home)
        predicted_away = round(adjusted_away)
        
        # Make sure home wins by at least 1
        if predicted_home <= predicted_away:
            predicted_home = predicted_away + 1
            
    else:  # Away team wins
        predicted_home = round(adjusted_home)
        predicted_away = round(adjusted_away)
        
        # Make sure away wins by at least 1
        if predicted_away <= predicted_home:
            predicted_away = predicted_home + 1
    
    # Keep scores realistic (2-7 goals typically)
    predicted_home = max(2, min(7, predicted_home))
    predicted_away = max(2, min(7, predicted_away))
    
    # Final check - ensure winner still wins
    if predicted_winner == home_team and predicted_home <= predicted_away:
        predicted_home = predicted_away + 1
    elif predicted_winner == away_team and predicted_away <= predicted_home:
        predicted_away = predicted_home + 1
    
    return predicted_home, predicted_away

def get_excel_prediction_from_sheet(home_team, away_team, game_date, predictions):
    """
    Get Excel prediction directly from the NHL HomeIce Model sheet
    """
    try:
        # Convert game_date to datetime if it's not already
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        # Find matching game
        game_match = predictions[
            (predictions['Home'] == home_team) & 
            (predictions['Visitor'] == away_team) &
            (predictions['Date'].dt.date == game_date.date())
        ]
        
        if len(game_match) > 0:
            game = game_match.iloc[0]
            return {
                'predicted_winner': game['Predicted Winner'],
                'homeice_diff': game['HomeIce Differential'],
                'strength_of_win': game.get('Strength of Win', 0),
                'has_excel': True
            }
    except Exception as e:
        st.sidebar.warning(f"Error reading Excel prediction: {e}")
    
    return {'has_excel': False}

def calculate_prediction(home_team, away_team, standings, predictions=None, game_date=None):
    """
    Calculate prediction for a matchup
    If predictions sheet and game_date provided, read from Excel
    Otherwise calculate on the fly
    """
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    # Try to get Excel prediction first
    excel_pred = None
    if predictions is not None and game_date is not None:
        excel_pred = get_excel_prediction_from_sheet(home_team, away_team, game_date, predictions)
    
    # Use Excel prediction if available, otherwise calculate
    if excel_pred and excel_pred['has_excel']:
        predicted_winner = excel_pred['predicted_winner']
        homeice_diff = excel_pred['homeice_diff']
        win_prob = 0.5 + (abs(homeice_diff) / 12)
        win_prob = min(0.85, max(0.52, win_prob))
    else:
        # Calculate HomeIce Differential
        home_home_win_pct = home_row['HomeWin%']
        away_away_win_pct = away_row['AwayWin%']
        homeice_diff = (home_home_win_pct - away_away_win_pct) * 6
        
        # Determine winner based on differential
        if homeice_diff > 0:
            predicted_winner = home_team
        else:
            predicted_winner = away_team
        
        win_prob = 0.5 + (abs(homeice_diff) / 12)
        win_prob = min(0.85, max(0.52, win_prob))
    
    # Predict score based on winner and differential
    predicted_home, predicted_away = predict_score_from_excel(
        home_team, away_team, predicted_winner, homeice_diff, standings
    )
    
    # Calculate overtime probability
    ot_probability = calculate_overtime_probability(homeice_diff)
    is_overtime = ot_probability > 0.35  # Predict OT if >35% chance
    
    return {
        'predicted_winner': predicted_winner,
        'win_prob': win_prob,
        'predicted_home': predicted_home,
        'predicted_away': predicted_away,
        'homeice_diff': homeice_diff,
        'ot_probability': ot_probability,
        'is_overtime': is_overtime,
        'home_row': home_row,
        'away_row': away_row
    }

def get_ml_prediction(home_team, away_team, game_date, ml_predictions):
    """Get ML prediction for a specific game with OT support"""
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
        
        # Get OT prediction (handle various data types)
        ml_is_overtime = False
        ml_overtime_prob = 0.0
        
        if 'ml_is_overtime' in ml_game.index:
            ot_val = ml_game['ml_is_overtime']
            if pd.notna(ot_val):
                if isinstance(ot_val, bool):
                    ml_is_overtime = ot_val
                elif isinstance(ot_val, str):
                    ml_is_overtime = ot_val.lower() in ['true', '1', 'yes']
                else:
                    ml_is_overtime = bool(ot_val)
        
        if 'ml_overtime_probability' in ml_game.index:
            ot_prob = ml_game['ml_overtime_probability']
            if pd.notna(ot_prob):
                ml_overtime_prob = float(ot_prob)
        
        return {
            'ml_predicted_winner': ml_game['ml_predicted_winner'],
            'ml_home_win_prob': ml_game['ml_home_win_prob'],
            'ml_away_win_prob': ml_game['ml_away_win_prob'],
            'ml_confidence': ml_game['ml_confidence'],
            'ml_predicted_home': ml_game['ml_predicted_home_score'],
            'ml_predicted_away': ml_game['ml_predicted_away_score'],
            'ml_is_overtime': ml_is_overtime,
            'ml_overtime_probability': ml_overtime_prob,
            'has_ml': True
        }
    else:
        return {'has_ml': False}

def display_game_card(game, standings, predictions, ml_predictions):
    """Display a game card with both model predictions - Modern Layout"""
    home_team = game['Home']
    away_team = game['Visitor']
    game_time = game['Time']
    game_date = game['Date']
    
    # Get predictions from both models
    excel_prediction = calculate_prediction(home_team, away_team, standings, predictions, game_date)
    ml_prediction = get_ml_prediction(home_team, away_team, game_date, ml_predictions)
    
    home_row = excel_prediction['home_row']
    away_row = excel_prediction['away_row']
    
    # Create card with container
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Header with time - centered
        st.markdown(f"<div style='text-align: center; color: #000000; font-size: 1.1rem; margin-bottom: 1.5rem;'>🕐 {game_time}</div>", unsafe_allow_html=True)
        
        # Teams display - centered and responsive
        col1, col_vs, col2 = st.columns([2.5, 0.5, 2.5], gap="medium")
        
        with col1:
            st.markdown(f"<div class='team-display'>", unsafe_allow_html=True)
            st.markdown(f"### {away_team}")
            st.caption(f"**{int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])}** ({int(away_row['PTS'])} pts)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_vs:
            st.markdown("<div style='text-align: center; padding-top: 1.5rem; color: #000000;'><h2>@</h2></div>", unsafe_allow_html=True)
        
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
        ot_color, ot_bg = get_probability_color(excel_prediction['ot_probability'])
        
        # Predictions side by side - responsive columns
        col_excel, col_ml = st.columns(2, gap="large")
        
        with col_excel:
            with st.container():
                st.markdown("#### 📊 Excel Model")
                st.markdown(f'<div class="winner">🏆 {excel_prediction["predicted_winner"]}</div>', unsafe_allow_html=True)
                
                # Win Probability with color
                st.markdown(f"**Win Probability:**")
                st.markdown(f'<span class="prob-badge" style="color: {prob_color}; background: {prob_bg};">{excel_prediction["win_prob"]:.1%}</span>', unsafe_allow_html=True)
                
                # Predicted Score with OT indicator
                st.markdown(f"**Predicted Score:**")
                excel_ot_text = " (OT/SO)" if excel_prediction['is_overtime'] else ""
                st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding: 0.5rem;'>{away_team} <strong>{excel_prediction['predicted_away']}</strong> - <strong>{excel_prediction['predicted_home']}</strong> {home_team}{excel_ot_text}</div>", unsafe_allow_html=True)
                
                # HomeIce Differential with color
                st.markdown(f"**HomeIce Diff:**")
                st.markdown(f'<span class="diff-badge" style="color: {diff_color}; background: {diff_bg};">{excel_prediction["homeice_diff"]:+.3f}</span>', unsafe_allow_html=True)
                
                # OT Probability
                st.markdown(f"**OT Probability:**")
                st.markdown(f'<span class="diff-badge" style="color: {ot_color}; background: {ot_bg};">⏱️ {excel_prediction["ot_probability"]:.1%}</span>', unsafe_allow_html=True)
        
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
                    
                    # Predicted Score with OT indicator
                    st.markdown(f"**Predicted Score:**")
                    ml_ot_text = " (OT/SO)" if ml_prediction.get('ml_is_overtime', False) else ""
                    st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding: 0.5rem;'>{away_team} <strong>{ml_prediction['ml_predicted_away']}</strong> - <strong>{ml_prediction['ml_predicted_home']}</strong> {home_team}{ml_ot_text}</div>", unsafe_allow_html=True)
                    
                    # OT Probability (if overtime predicted or significant probability)
                    ml_ot_prob = ml_prediction.get('ml_overtime_probability', 0.0)
                    if ml_ot_prob > 0.3:  # Show if >30% chance
                        st.markdown(f"**OT Probability:**")
                        ot_prob_color, ot_prob_bg = get_probability_color(ml_ot_prob)
                        st.markdown(f'<span class="diff-badge" style="color: {ot_prob_color}; background: {ot_prob_bg};">⏱️ {ml_ot_prob:.1%}</span>', unsafe_allow_html=True)
                    
                    # ML Confidence with color
                    ml_conf_color, ml_conf_bg = get_probability_color(ml_prediction['ml_confidence'])
                    st.markdown(f"**Confidence:**")
                    st.markdown(f'<span class="diff-badge" style="color: {ml_conf_color}; background: {ml_conf_bg};">{ml_prediction["ml_confidence"]:.1%}</span>', unsafe_allow_html=True)
                else:
                    st.info("No ML prediction available")
        
        # Agreement indicator - centered
        if ml_prediction['has_ml']:
            st.markdown("<br>", unsafe_allow_html=True)
            col_agreement, _, _ = st.columns([1, 2, 1])
            with col_agreement:
                agreements = []
                if excel_prediction['predicted_winner'] == ml_prediction['ml_predicted_winner']:
                    agreements.append("✅ Winner")
                else:
                    agreements.append("⚠️ Winner")
                
                # Check OT agreement
                both_ot = excel_prediction['is_overtime'] and ml_prediction.get('ml_is_overtime', False)
                neither_ot = not excel_prediction['is_overtime'] and not ml_prediction.get('ml_is_overtime', False)
                if both_ot or neither_ot:
                    agreements.append("✅ OT")
                else:
                    agreements.append("⚠️ OT")
                
                st.info(" | ".join(agreements))
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

def display_custom_matchup(home_team, away_team, standings, predictions, ml_predictions):
    """Display custom matchup with predictions - Modern Layout"""
    excel_prediction = calculate_prediction(home_team, away_team, standings, predictions, None)
    
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
            st.markdown("<div style='text-align: center; padding-top: 1.5rem; color: #000000;'><h2>@</h2></div>", unsafe_allow_html=True)
        
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
        ot_color, ot_bg = get_probability_color(excel_prediction['ot_probability'])
        
        # Show predictions side by side
        col_excel, col_ml = st.columns(2, gap="large")
        
        with col_excel:
            with st.container():
                st.markdown("#### 📊 Excel Model")
                st.markdown(f'<div class="winner">🏆 {excel_prediction["predicted_winner"]}</div>', unsafe_allow_html=True)
                
                # Win Probability with color
                st.markdown(f"**Win Probability:**")
                st.markdown(f'<span class="prob-badge" style="color: {prob_color}; background: {prob_bg};">{excel_prediction["win_prob"]:.1%}</span>', unsafe_allow_html=True)
                
                # Predicted Score with OT indicator
                st.markdown(f"**Predicted Score:**")
                excel_ot_text = " (OT/SO)" if excel_prediction['is_overtime'] else ""
                st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding: 0.5rem;'>{away_team} <strong>{excel_prediction['predicted_away']}</strong> - <strong>{excel_prediction['predicted_home']}</strong> {home_team}{excel_ot_text}</div>", unsafe_allow_html=True)
                
                # HomeIce Differential with color
                st.markdown(f"**HomeIce Diff:**")
                st.markdown(f'<span class="diff-badge" style="color: {diff_color}; background: {diff_bg};">{excel_prediction["homeice_diff"]:+.3f}</span>', unsafe_allow_html=True)
                
                # OT Probability
                st.markdown(f"**OT Probability:**")
                st.markdown(f'<span class="diff-badge" style="color: {ot_color}; background: {ot_bg};">⏱️ {excel_prediction["ot_probability"]:.1%}</span>', unsafe_allow_html=True)
        
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
                    
                    # Predicted Score with OT indicator
                    st.markdown(f"**Predicted Score:**")
                    ml_ot_text = " (OT/SO)" if ml_prediction.get('ml_is_overtime', False) else ""
                    st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding: 0.5rem;'>{away_team} <strong>{ml_prediction['ml_predicted_away']}</strong> - <strong>{ml_prediction['ml_predicted_home']}</strong> {home_team}{ml_ot_text}</div>", unsafe_allow_html=True)
                    
                    # OT Probability (if overtime predicted or significant probability)
                    ml_ot_prob = ml_prediction.get('ml_overtime_probability', 0.0)
                    if ml_ot_prob > 0.3:  # Show if >30% chance
                        st.markdown(f"**OT Probability:**")
                        ot_prob_color, ot_prob_bg = get_probability_color(ml_ot_prob)
                        st.markdown(f'<span class="diff-badge" style="color: {ot_prob_color}; background: {ot_prob_bg};">⏱️ {ml_ot_prob:.1%}</span>', unsafe_allow_html=True)
                    
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
        
        # Center the table
        col_table, _, _ = st.columns([1, 0.3, 1])
        with col_table:
            stats_data = {
                "Stat": ["Record", "Points", "Win %", "Goals Per Game", "Goals Allowed"],
                away_team: [
                    f"{int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])}",
                    int(away_row['PTS']),
                    f"{away_row['AwayWin%']:.1%}",
                    f"{away_row['Away Goals per Game']:.2f}",
                    f"{away_row['Away Goals Against']:.2f}"
                ],
                home_team: [
                    f"{int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['OTL'])}",
                    int(home_row['PTS']),
                    f"{home_row['HomeWin%']:.1%}",
                    f"{home_row['Home Goals per Game']:.2f}",
                    f"{home_row['Home Goals Against']:.2f}"
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
    st.markdown(f"<div style='text-align: center; color: #000000; margin-bottom: 2rem;'>📅 {datetime.now().strftime('%A, %B %d, %Y')}</div>", unsafe_allow_html=True)
    
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
        pages = ["Today's Games", "Custom Matchup", "Model Performance", "Model Comparison"]
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
            # Center the warning
            col_warn, _, _ = st.columns([1, 1, 1])
            with col_warn:
                st.warning("⚠️ No games scheduled for today")
            
            # Show upcoming games
            future_games = predictions[predictions['Date'].dt.date > today].copy()
            if len(future_games) > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📅 Next Upcoming Games")
                future_games = future_games.sort_values('Date').head(5)
                
                # Center upcoming games list
                col_upcoming, _, _ = st.columns([1, 1, 1])
                with col_upcoming:
                    for idx, game in future_games.iterrows():
                        date_str = game['Date'].strftime('%A, %B %d')
                        st.info(f"**{game['Visitor']} @ {game['Home']}** - {date_str}")
        else:
            st.subheader(f"🏒 Today's Games ({len(todays_games)} matchups)")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Sort by time
            todays_games = todays_games.sort_values('Time')
            
            # Display each game - wider for desktop
            for idx, game in todays_games.iterrows():
                col_game, _, _ = st.columns([1, 0.2, 1])
                with col_game:
                    display_game_card(game, standings, predictions, ml_predictions)
    
    # CUSTOM MATCHUP PAGE
    elif page == "Custom Matchup":
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⚙️ Custom Matchup Prediction")
        st.markdown("<br>", unsafe_allow_html=True)
        
        teams = sorted(standings['Team'].tolist())
        
        # Center the team selectors - better desktop layout
        col_outer1, col_inner, col_outer2 = st.columns([1, 2, 1])
        with col_inner:
            col1, col_vs, col2 = st.columns([2.5, 0.5, 2.5], gap="medium")
            
            with col1:
                away_team = st.selectbox("**Away Team**", [""] + teams, key="away")
            
            with col_vs:
                st.markdown("<div style='text-align: center; padding-top: 2rem; color: #000000;'><h3>@</h3></div>", unsafe_allow_html=True)
            
            with col2:
                home_team = st.selectbox("**Home Team**", [""] + teams, key="home")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Center the button
            col_btn, _, _ = st.columns([1, 1, 1])
            with col_btn:
                if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
                    if not away_team or not home_team:
                        st.warning("⚠️ Please select both teams")
                    elif away_team == home_team:
                        st.error("❌ Please select different teams")
                    else:
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_result, _, _ = st.columns([1, 0.2, 1])
                        with col_result:
                            display_custom_matchup(home_team, away_team, standings, predictions, ml_predictions)
    
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
            col_info, _, _ = st.columns([2, 1, 2])
            with col_info:
                st.info("No completed games yet for Excel model")
        else:
            completed = completed.sort_values('Date')
            
            # Overall stats - centered
            total_games = len(completed)
            correct_games = (completed[correct_col] == 'YES').sum()
            overall_accuracy = (correct_games / total_games * 100)
            
            col_metrics, _, _ = st.columns([1, 0.3, 1])
            with col_metrics:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("#### 📈 Overall Results")
                
                col1, col2, col3, col4 = st.columns(4, gap="small")
                
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
                
                col_last20, _, _ = st.columns([1, 0.3, 1])
                with col_last20:
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
            col_info, _, _ = st.columns([2, 1, 2])
            with col_info:
                st.info("⚠️ ML Model predictions not available. Upload an Excel file with 'ML Prediction Model' sheet to see ML performance.")
        else:
            ml_completed = ml_predictions[ml_predictions['ml_correct'].notna()].copy()
            
            if len(ml_completed) == 0:
                col_info, _, _ = st.columns([2, 1, 2])
                with col_info:
                    st.info("No completed games yet for ML model")
            else:
                ml_completed = ml_completed.sort_values('date')
                
                # Overall stats - centered
                ml_total = len(ml_completed)
                ml_correct = (ml_completed['ml_correct'] == 1).sum()
                ml_accuracy = (ml_correct / ml_total * 100)
                
                col_ml_metrics, _, _ = st.columns([1, 0.3, 1])
                with col_ml_metrics:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("#### 📈 Overall Results")
                    
                    col1, col2, col3, col4 = st.columns(4, gap="small")
                    
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
                    correct_conf = ml_completed[ml_completed['ml_correct'] == 1]['ml_confidence'].mean()
                    incorrect_conf = ml_completed[ml_completed['ml_correct'] == 0]['ml_confidence'].mean()
                    
                    col_conf, _, _ = st.columns([1, 0.3, 1])
                    with col_conf:
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
            col_info, _, _ = st.columns([2, 1, 2])
            with col_info:
                st.info("No completed games to compare yet")
        else:
            ml_completed = ml_completed.sort_values('date')
            
            # Overall comparison - centered
            ml_total = len(ml_completed)
            ml_correct = (ml_completed['ml_correct'] == 1).sum()
            excel_correct = (ml_completed['excel_correct'] == 1).sum()
            
            ml_accuracy = (ml_correct / ml_total * 100)
            excel_accuracy = (excel_correct / ml_total * 100)
            
            col_comparison, _, _ = st.columns([1, 0.3, 1])
            with col_comparison:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Head-to-Head Comparison")
                st.caption(f"Based on {ml_total} completed games")
                
                col1, col2, col3 = st.columns(3, gap="medium")
                
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
            both_correct = ((ml_completed['ml_correct'] == 1) & (ml_completed['excel_correct'] == 1)).sum()
            both_wrong = ((ml_completed['ml_correct'] == 0) & (ml_completed['excel_correct'] == 0)).sum()
            ml_only = ((ml_completed['ml_correct'] == 1) & (ml_completed['excel_correct'] == 0)).sum()
            excel_only = ((ml_completed['ml_correct'] == 0) & (ml_completed['excel_correct'] == 1)).sum()
            
            col_agreement, _, _ = st.columns([1, 0.3, 1])
            with col_agreement:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### 🤝 Model Agreement")
                
                col1, col2, col3, col4 = st.columns(4, gap="small")
                
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
                excel_result = "✅" if game['excel_correct'] == 1 else "❌"
                ml_result = "✅" if game['ml_correct'] == 1 else "❌"
                
                display_data.append({
                    "Date": game['date'].strftime('%Y-%m-%d'),
                    "Matchup": f"{game['away_team']} @ {game['home_team']}",
                    "Winner": game['actual_winner'],
                    "Excel": f"{excel_result} {game['excel_predicted_winner']}",
                    "ML": f"{ml_result} {game['ml_predicted_winner']}"
                })
            
            # Center the table
            col_table, _, _ = st.columns([1, 0.3, 1])
            with col_table:
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
        - 📊 Excel: HomeIce Differential with score & OT predictions
        - 🤖 ML: Random Forest machine learning model
        
        Compare predictions and performance across both approaches!
        """)
    else:
        st.sidebar.info("""
        **NHL Prediction Model 2025-26**
        
        📊 **Excel Model**
        Using HomeIce Differential and team statistics to predict:
        - Game winner
        - Final score
        - Overtime likelihood
        
        💡 Add 'ML Prediction Model' sheet to Excel file to enable ML predictions!
        """)

if __name__ == "__main__":
    main()
