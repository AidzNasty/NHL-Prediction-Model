#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHL Prediction Model - Enhanced Web App with BOTH Excel AND ML Score/OT Predictions
Both models now show: Winner, Score, Overtime Prediction
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

# Custom CSS - Compact, Responsive Design
st.markdown("""
    <style>
    /* Main App Styling */
    .main {
        background: #f5f5f5;
        padding: 1rem;
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
        font-size: 2rem;
    }
    h2 {
        font-size: 1.5rem;
    }
    h3, h4 {
        font-size: 1.2rem;
    }
    
    /* Centered Container */
    .centered-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Game Card - More Compact */
    .game-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }
    
    /* Winner Styling */
    .winner {
        color: #2e7d32;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        padding: 0.5rem;
        background: #e8f5e9;
        border-radius: 6px;
        margin: 0.3rem 0;
        border: 2px solid #4caf50;
    }
    
    /* Score Display */
    .score-display {
        text-align: center;
        font-size: 1.1rem;
        padding: 0.4rem;
        background: #fafafa;
        border-radius: 6px;
        margin: 0.3rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* OT Badge */
    .ot-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-left: 0.3rem;
    }
    .ot-yes {
        background: #fff3e0;
        color: #e65100;
        border: 1px solid #ffb74d;
    }
    .ot-maybe {
        background: #e3f2fd;
        color: #1976d2;
        border: 1px solid #64b5f6;
    }
    .ot-no {
        background: #f5f5f5;
        color: #616161;
        border: 1px solid #bdbdbd;
    }
    
    /* Metric Text - Smaller */
    .metric-text {
        font-size: 0.9rem;
        margin: 0.2rem 0;
    }
    
    /* Badge Styling - Smaller */
    .prob-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Team Display - More Compact */
    .team-display {
        text-align: center;
        padding: 0.5rem;
    }
    
    /* Compact Spacing */
    .compact-section {
        margin: 0.5rem 0;
    }
    
    /* All text black */
    body, p, span, div, label, .stText, .stMarkdown {
        color: #000000 !important;
    }
    
    /* Responsive - Better mobile */
    @media (max-width: 768px) {
        .game-card {
            padding: 1rem;
        }
        h1 {
            font-size: 1.5rem;
        }
        h4 {
            font-size: 1rem;
        }
        .winner {
            font-size: 1.1rem;
        }
        .score-display {
            font-size: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Constants
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
    return EXCEL_FILE_ALTERNATIVES[0]

EXCEL_FILE = find_excel_file()

def get_probability_color(probability):
    """Convert probability to color gradient"""
    normalized = (probability - 0.5) / 0.5
    normalized = max(0, min(1, normalized))
    
    if normalized < 0.5:
        t = normalized * 2
        r = int(211 + (251 - 211) * t)
        g = int(47 + (192 - 47) * t)
        b = int(47 + (45 - 47) * t)
    else:
        t = (normalized - 0.5) * 2
        r = int(251 + (56 - 251) * t)
        g = int(192 + (142 - 192) * t)
        b = int(45 + (60 - 45) * t)
    
    color = f"#{r:02x}{g:02x}{b:02x}"
    bg_color = f"rgba({r}, {g}, {b}, 0.15)"
    return color, bg_color

def convert_percentage_string(pct_str):
    """Convert percentage string to float"""
    try:
        if pd.isna(pct_str) or pct_str == 'nan%':
            return 0.0
        if isinstance(pct_str, str):
            return float(pct_str.strip('%')) / 100.0
        return float(pct_str)
    except:
        return 0.0

@st.cache_data(ttl=3600)
def load_data():
    """Load Excel data with caching"""
    load_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    import os
    st.sidebar.caption(f"📁 Using: {EXCEL_FILE}")
    st.sidebar.caption(f"📁 Exists: {os.path.exists(EXCEL_FILE)}")
    
    try:
        # Load NHL HomeIce Model sheet
        excel_predictions_raw = pd.read_excel(EXCEL_FILE, sheet_name='NHL HomeIce Model', header=None)
        header_row = excel_predictions_raw.iloc[0].values
        predictions_data = excel_predictions_raw.iloc[1:].reset_index(drop=True)
        predictions_data.columns = header_row
        predictions = predictions_data.iloc[:, 3:].copy()
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        # Load ML predictions
        ml_predictions = None
        try:
            st.sidebar.info("🔄 Loading ML Model...")
            ml_predictions = pd.read_excel(EXCEL_FILE, sheet_name='ML Prediction Model', header=0)
            st.sidebar.info(f"📊 Found {len(ml_predictions)} ML predictions")
            
            # Find date column
            date_col = None
            for col in ['game_date', 'date', 'Date']:
                if col in ml_predictions.columns:
                    date_col = col
                    break
            
            if date_col:
                ml_predictions['date'] = pd.to_datetime(ml_predictions[date_col])
            
            # Rename columns
            rename_map = {
                'ml_winner': 'ml_predicted_winner',
                'ml_home_score': 'ml_predicted_home_score',
                'ml_away_score': 'ml_predicted_away_score',
                'ml_ot': 'ml_is_overtime',
                'ml_ot_prob': 'ml_overtime_probability',
                'excel_winner': 'excel_predicted_winner'
            }
            ml_predictions = ml_predictions.rename(columns={k: v for k, v in rename_map.items() if k in ml_predictions.columns})
            
            # Convert percentages
            if 'ml_confidence' in ml_predictions.columns:
                ml_predictions['ml_confidence'] = ml_predictions['ml_confidence'].apply(convert_percentage_string)
            else:
                ml_predictions['ml_confidence'] = 0.5
                
            if 'ml_overtime_probability' in ml_predictions.columns:
                ml_predictions['ml_overtime_probability'] = ml_predictions['ml_overtime_probability'].apply(convert_percentage_string)
            else:
                ml_predictions['ml_overtime_probability'] = 0.0
            
            # Convert OT to boolean
            if 'ml_is_overtime' in ml_predictions.columns:
                ml_predictions['ml_is_overtime'] = ml_predictions['ml_is_overtime'].apply(
                    lambda x: str(x).upper() in ['YES', 'TRUE', '1'] if pd.notna(x) else False
                )
            else:
                ml_predictions['ml_is_overtime'] = False
            
            # Calculate win probabilities
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
            
            # Convert correct columns
            for col in ['ml_correct', 'excel_correct']:
                if col in ml_predictions.columns:
                    ml_predictions[col] = ml_predictions[col].apply(
                        lambda x: 1 if str(x).upper() == 'YES' else (0 if str(x).upper() == 'NO' else np.nan)
                    )
            
            st.sidebar.success(f"✅ ML Model loaded ({len(ml_predictions)} games)")
            
        except Exception as ml_error:
            st.sidebar.error("❌ ML Model failed to load")
            st.sidebar.caption(f"Error: {str(ml_error)}")
            ml_predictions = pd.DataFrame()
        
        return predictions, standings, ml_predictions, load_timestamp
    except Exception as e:
        st.error(f"❌ Error loading Excel: {e}")
        return None, None, None, None

def calculate_overtime_probability(homeice_diff):
    """Calculate OT probability from HomeIce Differential"""
    abs_diff = abs(homeice_diff)
    
    if abs_diff < 0.2:
        ot_prob = 0.40 + (0.2 - abs_diff) * 0.5
    elif abs_diff < 0.5:
        ot_prob = 0.25 + (0.5 - abs_diff) * 0.5
    elif abs_diff < 1.0:
        ot_prob = 0.10 + (1.0 - abs_diff) * 0.3
    else:
        ot_prob = max(0.05, 0.10 - (abs_diff - 1.0) * 0.05)
    
    return max(0.05, min(0.50, ot_prob))

def predict_score_from_excel(home_team, away_team, predicted_winner, homeice_diff, standings):
    """Predict game score based on Excel model"""
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    home_goals_for = home_row['Home Goals per Game']
    home_goals_against = home_row['Home Goals Against']
    away_goals_for = away_row['Away Goals per Game']
    away_goals_against = away_row['Away Goals Against']
    
    expected_home = (home_goals_for + away_goals_against) / 2
    expected_away = (away_goals_for + home_goals_against) / 2
    
    home_adjustment = homeice_diff * 0.5
    away_adjustment = -homeice_diff * 0.5
    
    adjusted_home = expected_home + home_adjustment
    adjusted_away = expected_away + away_adjustment
    
    predicted_home = round(adjusted_home)
    predicted_away = round(adjusted_away)
    
    if predicted_winner == home_team and predicted_home <= predicted_away:
        predicted_home = predicted_away + 1
    elif predicted_winner == away_team and predicted_away <= predicted_home:
        predicted_away = predicted_home + 1
    
    predicted_home = max(2, min(7, predicted_home))
    predicted_away = max(2, min(7, predicted_away))
    
    if predicted_winner == home_team and predicted_home <= predicted_away:
        predicted_home = predicted_away + 1
    elif predicted_winner == away_team and predicted_away <= predicted_home:
        predicted_away = predicted_home + 1
    
    return predicted_home, predicted_away

def get_excel_prediction_from_sheet(home_team, away_team, game_date, predictions):
    """Get Excel prediction from HomeIce Model sheet"""
    try:
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
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
                'has_excel': True
            }
    except:
        pass
    
    return {'has_excel': False}

def calculate_prediction(home_team, away_team, standings, predictions=None, game_date=None):
    """Calculate Excel prediction"""
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    excel_pred = None
    if predictions is not None and game_date is not None:
        excel_pred = get_excel_prediction_from_sheet(home_team, away_team, game_date, predictions)
    
    if excel_pred and excel_pred['has_excel']:
        predicted_winner = excel_pred['predicted_winner']
        homeice_diff = excel_pred['homeice_diff']
    else:
        home_home_win_pct = home_row['HomeWin%']
        away_away_win_pct = away_row['AwayWin%']
        homeice_diff = (home_home_win_pct - away_away_win_pct) * 6
        predicted_winner = home_team if homeice_diff > 0 else away_team
    
    win_prob = 0.5 + (abs(homeice_diff) / 12)
    win_prob = min(0.85, max(0.52, win_prob))
    
    predicted_home, predicted_away = predict_score_from_excel(
        home_team, away_team, predicted_winner, homeice_diff, standings
    )
    
    ot_probability = calculate_overtime_probability(homeice_diff)
    is_overtime = ot_probability > 0.35
    
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
    """Get ML prediction for a specific game"""
    if ml_predictions is None or len(ml_predictions) == 0:
        return {'has_ml': False}
    
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    
    ml_game = ml_predictions[
        (ml_predictions['home_team'] == home_team) & 
        (ml_predictions['away_team'] == away_team) &
        (ml_predictions['date'].dt.date == game_date.date())
    ]
    
    if len(ml_game) > 0:
        ml_game = ml_game.iloc[0]
        
        # Get scores with proper conversion
        ml_home_score = ml_game.get('ml_predicted_home_score', 3)
        ml_away_score = ml_game.get('ml_predicted_away_score', 2)
        
        # Convert to int if not NaN
        try:
            ml_home_score = int(float(ml_home_score)) if pd.notna(ml_home_score) else 3
            ml_away_score = int(float(ml_away_score)) if pd.notna(ml_away_score) else 2
        except:
            ml_home_score = 3
            ml_away_score = 2
        
        # Get OT prediction
        ml_is_overtime = False
        if 'ml_is_overtime' in ml_game.index:
            ot_val = ml_game['ml_is_overtime']
            if pd.notna(ot_val):
                ml_is_overtime = bool(ot_val)
        
        ml_overtime_prob = 0.0
        if 'ml_overtime_probability' in ml_game.index:
            ot_prob = ml_game['ml_overtime_probability']
            if pd.notna(ot_prob):
                ml_overtime_prob = float(ot_prob)
        
        return {
            'ml_predicted_winner': ml_game['ml_predicted_winner'],
            'ml_home_win_prob': ml_game['ml_home_win_prob'],
            'ml_away_win_prob': ml_game['ml_away_win_prob'],
            'ml_confidence': ml_game['ml_confidence'],
            'ml_predicted_home': ml_home_score,
            'ml_predicted_away': ml_away_score,
            'ml_is_overtime': ml_is_overtime,
            'ml_overtime_probability': ml_overtime_prob,
            'has_ml': True
        }
    else:
        return {'has_ml': False}

def get_ot_badge_html(ot_prob, is_ot):
    """Generate OT badge HTML"""
    if ot_prob > 0.40 or is_ot:
        badge_class = "ot-yes"
        text = f"⏱️ OT Likely ({ot_prob:.0%})"
    elif ot_prob > 0.25:
        badge_class = "ot-maybe"
        text = f"⏱️ OT Possible ({ot_prob:.0%})"
    else:
        badge_class = "ot-no"
        text = f"Regulation ({100-ot_prob*100:.0f}%)"
    
    return f'<span class="ot-badge {badge_class}">{text}</span>'

def display_game_card(game, standings, predictions, ml_predictions):
    """Display game card with both model predictions - COMPACT"""
    home_team = game['Home']
    away_team = game['Visitor']
    game_time = game['Time']
    game_date = game['Date']
    
    excel_prediction = calculate_prediction(home_team, away_team, standings, predictions, game_date)
    ml_prediction = get_ml_prediction(home_team, away_team, game_date, ml_predictions)
    
    home_row = excel_prediction['home_row']
    away_row = excel_prediction['away_row']
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Header - Compact
        st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.95rem; margin-bottom: 0.8rem;'>🕐 {game_time}</div>", unsafe_allow_html=True)
        
        # Teams - Compact
        col1, col_vs, col2 = st.columns([2, 0.3, 2])
        
        with col1:
            st.markdown(f"<div class='team-display'><h4 style='margin: 0;'>{away_team}</h4>", unsafe_allow_html=True)
            st.caption(f"{int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])} ({int(away_row['PTS'])} pts)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_vs:
            st.markdown("<div style='text-align: center; padding-top: 0.8rem;'><strong>@</strong></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div class='team-display'><h4 style='margin: 0;'>{home_team}</h4>", unsafe_allow_html=True)
            st.caption(f"{int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['OTL'])} ({int(home_row['PTS'])} pts)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Predictions side-by-side - COMPACT
        col_excel, col_ml = st.columns(2)
        
        with col_excel:
            st.markdown("<div class='compact-section'>", unsafe_allow_html=True)
            st.markdown("**📊 Excel Model**")
            st.markdown(f'<div class="winner">🏆 {excel_prediction["predicted_winner"]}</div>', unsafe_allow_html=True)
            
            # Score
            st.markdown(f'<div class="score-display"><strong>{excel_prediction["predicted_away"]}-{excel_prediction["predicted_home"]}</strong></div>', unsafe_allow_html=True)
            
            # OT Prediction
            ot_badge = get_ot_badge_html(excel_prediction['ot_probability'], excel_prediction['is_overtime'])
            st.markdown(ot_badge, unsafe_allow_html=True)
            
            # Compact metrics
            prob_color, prob_bg = get_probability_color(excel_prediction['win_prob'])
            st.markdown(f'<div class="metric-text">Win Prob: <span class="prob-badge" style="color: {prob_color}; background: {prob_bg};">{excel_prediction["win_prob"]:.0%}</span></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_ml:
            st.markdown("<div class='compact-section'>", unsafe_allow_html=True)
            st.markdown("**🤖 ML Model**")
            if ml_prediction['has_ml']:
                st.markdown(f'<div class="winner">🏆 {ml_prediction["ml_predicted_winner"]}</div>', unsafe_allow_html=True)
                
                # Score
                st.markdown(f'<div class="score-display"><strong>{ml_prediction["ml_predicted_away"]}-{ml_prediction["ml_predicted_home"]}</strong></div>', unsafe_allow_html=True)
                
                # OT Prediction
                ml_ot_badge = get_ot_badge_html(ml_prediction['ml_overtime_probability'], ml_prediction['ml_is_overtime'])
                st.markdown(ml_ot_badge, unsafe_allow_html=True)
                
                # Compact metrics
                ml_win_prob = ml_prediction['ml_home_win_prob'] if ml_prediction['ml_predicted_winner'] == home_team else ml_prediction['ml_away_win_prob']
                ml_prob_color, ml_prob_bg = get_probability_color(ml_win_prob)
                st.markdown(f'<div class="metric-text">Win Prob: <span class="prob-badge" style="color: {ml_prob_color}; background: {ml_prob_bg};">{ml_win_prob:.0%}</span></div>', unsafe_allow_html=True)
            else:
                st.info("No ML prediction")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Agreement - Compact
        if ml_prediction['has_ml']:
            agreements = []
            if excel_prediction['predicted_winner'] == ml_prediction['ml_predicted_winner']:
                agreements.append("✅ Winner")
            else:
                agreements.append("⚠️ Different Winners")
            
            both_ot = excel_prediction['is_overtime'] and ml_prediction['ml_is_overtime']
            neither_ot = not excel_prediction['is_overtime'] and not ml_prediction['ml_is_overtime']
            if both_ot or neither_ot:
                agreements.append("✅ OT")
            
            st.caption(" | ".join(agreements))
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_custom_matchup(home_team, away_team, standings, predictions, ml_predictions):
    """Display custom matchup - same format as game card"""
    excel_prediction = calculate_prediction(home_team, away_team, standings, predictions, None)
    today = datetime.now()
    ml_prediction = get_ml_prediction(home_team, away_team, today, ml_predictions)
    
    home_row = excel_prediction['home_row']
    away_row = excel_prediction['away_row']
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Teams
        col1, col_vs, col2 = st.columns([2, 0.3, 2])
        
        with col1:
            st.markdown(f"<div class='team-display'><h4 style='margin: 0;'>{away_team}</h4>", unsafe_allow_html=True)
            st.caption(f"{int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])} ({int(away_row['PTS'])} pts)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_vs:
            st.markdown("<div style='text-align: center; padding-top: 0.8rem;'><strong>@</strong></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div class='team-display'><h4 style='margin: 0;'>{home_team}</h4>", unsafe_allow_html=True)
            st.caption(f"{int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['OTL'])} ({int(home_row['PTS'])} pts)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Predictions
        col_excel, col_ml = st.columns(2)
        
        with col_excel:
            st.markdown("**📊 Excel Model**")
            st.markdown(f'<div class="winner">🏆 {excel_prediction["predicted_winner"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-display"><strong>{excel_prediction["predicted_away"]}-{excel_prediction["predicted_home"]}</strong></div>', unsafe_allow_html=True)
            ot_badge = get_ot_badge_html(excel_prediction['ot_probability'], excel_prediction['is_overtime'])
            st.markdown(ot_badge, unsafe_allow_html=True)
            prob_color, prob_bg = get_probability_color(excel_prediction['win_prob'])
            st.markdown(f'<div class="metric-text">Win Prob: <span class="prob-badge" style="color: {prob_color}; background: {prob_bg};">{excel_prediction["win_prob"]:.0%}</span></div>', unsafe_allow_html=True)
        
        with col_ml:
            st.markdown("**🤖 ML Model**")
            if ml_prediction['has_ml']:
                st.markdown(f'<div class="winner">🏆 {ml_prediction["ml_predicted_winner"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-display"><strong>{ml_prediction["ml_predicted_away"]}-{ml_prediction["ml_predicted_home"]}</strong></div>', unsafe_allow_html=True)
                ml_ot_badge = get_ot_badge_html(ml_prediction['ml_overtime_probability'], ml_prediction['ml_is_overtime'])
                st.markdown(ml_ot_badge, unsafe_allow_html=True)
                ml_win_prob = ml_prediction['ml_home_win_prob'] if ml_prediction['ml_predicted_winner'] == home_team else ml_prediction['ml_away_win_prob']
                ml_prob_color, ml_prob_bg = get_probability_color(ml_win_prob)
                st.markdown(f'<div class="metric-text">Win Prob: <span class="prob-badge" style="color: {ml_prob_color}; background: {ml_prob_bg};">{ml_win_prob:.0%}</span></div>', unsafe_allow_html=True)
            else:
                st.info("No ML prediction")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    predictions, standings, ml_predictions, load_timestamp = load_data()
    
    if predictions is None or standings is None:
        return
    
    st.markdown("<div class='centered-container'>", unsafe_allow_html=True)
    st.title("🏒 NHL Prediction Model 2025-26")
    st.markdown(f"<div style='text-align: center; color: #666; margin-bottom: 1.5rem;'>📅 {datetime.now().strftime('%A, %B %d, %Y')}</div>", unsafe_allow_html=True)
    
    st.sidebar.title("🧭 Navigation")
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.caption(f"📊 Loaded: {load_timestamp}")
    st.sidebar.markdown("---")
    
    has_ml_data = ml_predictions is not None and len(ml_predictions) > 0
    pages = ["Today's Games", "Custom Matchup", "Model Performance", "Model Comparison"] if has_ml_data else ["Today's Games", "Custom Matchup", "Model Performance"]
    page = st.sidebar.radio("Select Page", pages)
    
    if page == "Today's Games":
        today = datetime.now().date()
        todays_games = predictions[predictions['Date'].dt.date == today].copy()
        
        if len(todays_games) == 0:
            st.warning("⚠️ No games scheduled today")
            future_games = predictions[predictions['Date'].dt.date > today].head(5)
            if len(future_games) > 0:
                st.subheader("📅 Next Upcoming Games")
                for _, game in future_games.iterrows():
                    st.info(f"**{game['Visitor']} @ {game['Home']}** - {game['Date'].strftime('%A, %B %d')}")
        else:
            st.subheader(f"🏒 Today's Games ({len(todays_games)} matchups)")
            todays_games = todays_games.sort_values('Time')
            for _, game in todays_games.iterrows():
                display_game_card(game, standings, predictions, ml_predictions)
    
    elif page == "Custom Matchup":
        st.subheader("⚙️ Custom Matchup Prediction")
        teams = sorted(standings['Team'].tolist())
        
        col1, col_vs, col2 = st.columns([2, 0.3, 2])
        with col1:
            away_team = st.selectbox("Away Team", [""] + teams, key="away")
        with col_vs:
            st.markdown("<div style='text-align: center; padding-top: 2rem;'><strong>@</strong></div>", unsafe_allow_html=True)
        with col2:
            home_team = st.selectbox("Home Team", [""] + teams, key="home")
        
        if st.button("🎯 Generate Prediction", type="primary"):
            if away_team and home_team and away_team != home_team:
                display_custom_matchup(home_team, away_team, standings, predictions, ml_predictions)
            else:
                st.warning("Please select two different teams")
    
    elif page == "Model Performance":
        st.subheader("📊 Model Performance")
        
        # Excel Model
        st.markdown("### 📊 Excel Model")
        completed = predictions[predictions['Locked Correct'].isin(['YES', 'NO'])].copy()
        if len(completed) > 0:
            total = len(completed)
            correct = (completed['Locked Correct'] == 'YES').sum()
            accuracy = (correct / total * 100)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Games", total)
            col2.metric("Correct", correct)
            col3.metric("Accuracy", f"{accuracy:.1f}%")
        else:
            st.info("No completed games yet")
        
        # ML Model
        st.markdown("---")
        st.markdown("### 🤖 ML Model")
        if ml_predictions is not None and len(ml_predictions) > 0:
            ml_completed = ml_predictions[ml_predictions['ml_correct'].notna()].copy()
            if len(ml_completed) > 0:
                ml_total = len(ml_completed)
                ml_correct = (ml_completed['ml_correct'] == 1).sum()
                ml_accuracy = (ml_correct / ml_total * 100)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Games", ml_total)
                col2.metric("Correct", ml_correct)
                col3.metric("Accuracy", f"{ml_accuracy:.1f}%")
            else:
                st.info("No completed games yet")
        else:
            st.info("ML Model not available")
    
    elif page == "Model Comparison":
        st.subheader("🔄 Model Comparison")
        ml_completed = ml_predictions[ml_predictions['ml_correct'].notna()].copy()
        
        if len(ml_completed) > 0:
            ml_total = len(ml_completed)
            ml_correct = (ml_completed['ml_correct'] == 1).sum()
            excel_correct = (ml_completed['excel_correct'] == 1).sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Excel Accuracy", f"{excel_correct/ml_total*100:.1f}%")
            col2.metric("ML Accuracy", f"{ml_correct/ml_total*100:.1f}%")
            diff = (ml_correct - excel_correct) / ml_total * 100
            col3.metric("Difference", f"{diff:+.1f}%")
            
            st.markdown("---")
            both_correct = ((ml_completed['ml_correct'] == 1) & (ml_completed['excel_correct'] == 1)).sum()
            both_wrong = ((ml_completed['ml_correct'] == 0) & (ml_completed['excel_correct'] == 0)).sum()
            
            col1, col2 = st.columns(2)
            col1.metric("Both Correct", both_correct, f"{both_correct/ml_total*100:.1f}%")
            col2.metric("Both Wrong", both_wrong, f"{both_wrong/ml_total*100:.1f}%")
        else:
            st.info("No completed games to compare")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **NHL Prediction Model 2025-26**
    
    🔄 **Both Models Show:**
    - Winner prediction
    - Predicted score
    - Overtime likelihood
    - Win probability
    """)

if __name__ == "__main__":
    main()
