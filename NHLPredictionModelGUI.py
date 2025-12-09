#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHL Prediction Model with Player Predictions
Shows game predictions + top goal scorers/assist leaders/point getters
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

# Simplified CSS
st.markdown("""
    <style>
    .main {
        background: #ffffff;
        padding: 2rem;
    }
    
    h1, h2, h3 {
        color: #1a1a1a;
    }
    
    .game-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .prediction-box {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .player-box {
        background: #f0f8ff;
        padding: 0.8rem;
        border-radius: 6px;
        border: 1px solid #cce5ff;
        margin: 0.5rem 0;
    }
    
    .winner-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #28a745;
    }
    
    .score-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #495057;
    }
    
    .player-name {
        font-weight: 600;
        color: #0066cc;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
EXCEL_FILE = 'Aidan Conte NHL 2025-26 Prediction Model.xlsx'
EASTERN = pytz.timezone('America/New_York')

# Team abbreviation mapping
TEAM_ABBREV_MAP = {
    'ANA': 'Anaheim Ducks',
    'BOS': 'Boston Bruins',
    'BUF': 'Buffalo Sabres',
    'CAR': 'Carolina Hurricanes',
    'CBJ': 'Columbus Blue Jackets',
    'CGY': 'Calgary Flames',
    'CHI': 'Chicago Blackhawks',
    'COL': 'Colorado Avalanche',
    'DAL': 'Dallas Stars',
    'DET': 'Detroit Red Wings',
    'EDM': 'Edmonton Oilers',
    'FLA': 'Florida Panthers',
    'LAK': 'Los Angeles Kings',
    'MIN': 'Minnesota Wild',
    'MTL': 'Montreal Canadiens',
    'NJD': 'New Jersey Devils',
    'NSH': 'Nashville Predators',
    'NYI': 'New York Islanders',
    'NYR': 'New York Rangers',
    'OTT': 'Ottawa Senators',
    'PHI': 'Philadelphia Flyers',
    'PIT': 'Pittsburgh Penguins',
    'SEA': 'Seattle Kraken',
    'SJS': 'San Jose Sharks',
    'STL': 'St. Louis Blues',
    'TBL': 'Tampa Bay Lightning',
    'TOR': 'Toronto Maple Leafs',
    'UTA': 'Utah Mammoth',
    'VAN': 'Vancouver Canucks',
    'VEG': 'Vegas Golden Knights',
    'WPG': 'Winnipeg Jets',
    'WSH': 'Washington Capitals'
}

# Reverse mapping
TEAM_TO_ABBREV = {v: k for k, v in TEAM_ABBREV_MAP.items()}

@st.cache_data(ttl=3600)
def load_data():
    """Load Excel and player data"""
    try:
        # Load Excel predictions
        excel_predictions_raw = pd.read_excel(EXCEL_FILE, sheet_name='NHL HomeIce Model', header=None)
        header_row = excel_predictions_raw.iloc[0].values
        predictions_data = excel_predictions_raw.iloc[1:].reset_index(drop=True)
        predictions_data.columns = header_row
        predictions = predictions_data.iloc[:, 3:].copy()
        predictions['Date'] = pd.to_datetime(predictions['Date']).dt.normalize()
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        # Load player stats
        players = pd.read_excel(EXCEL_FILE, sheet_name='NHL2025-26PlayerStats')
        
        # Clean player data
        players = players[players['GP'] > 0].copy()  # Only players who have played
        players['GP'] = players['GP'].fillna(1)
        players['G'] = players['G'].fillna(0)
        players['A'] = players['A'].fillna(0)
        players['PTS'] = players['PTS'].fillna(0)
        players['SOG'] = players['SOG'].fillna(0)
        
        # Calculate per-game rates
        players['goals_pg'] = players['G'] / players['GP']
        players['assists_pg'] = players['A'] / players['GP']
        players['points_pg'] = players['PTS'] / players['GP']
        players['shots_pg'] = players['SOG'] / players['GP']
        
        # Map team abbreviations to full names
        players['Team_Full'] = players['Team'].map(TEAM_ABBREV_MAP)
        
        # Remove players with 2TM (played for multiple teams) or unmapped teams
        players = players[players['Team_Full'].notna()].copy()
        
        st.sidebar.success(f"Loaded {len(players)} players from {players['Team'].nunique()} teams")
        
        # Load ML predictions
        ml_predictions = None
        try:
            ml_predictions = pd.read_excel(EXCEL_FILE, sheet_name='ML Prediction Model', header=0)
            
            if len(ml_predictions) > 0:
                if 'game_date' in ml_predictions.columns:
                    ml_predictions['date'] = pd.to_datetime(ml_predictions['game_date'], errors='coerce').dt.normalize()
                
                rename_map = {
                    'ml_winner': 'ml_predicted_winner',
                    'ml_home_score': 'ml_predicted_home_score',
                    'ml_away_score': 'ml_predicted_away_score',
                    'ml_ot': 'ml_is_overtime',
                    'ml_confidence': 'ml_confidence'
                }
                ml_predictions = ml_predictions.rename(columns={k: v for k, v in rename_map.items() if k in ml_predictions.columns})
                
                if 'ml_confidence' in ml_predictions.columns:
                    ml_predictions['ml_confidence'] = ml_predictions['ml_confidence'].apply(
                        lambda x: float(str(x).strip('%')) / 100.0 if isinstance(x, str) and '%' in str(x) else float(x) if pd.notna(x) else 0.5
                    )
                
                if 'ml_is_overtime' in ml_predictions.columns:
                    ml_predictions['ml_is_overtime'] = ml_predictions['ml_is_overtime'].apply(
                        lambda x: str(x).upper() in ['YES', 'TRUE', '1'] if pd.notna(x) else False
                    )
                
                st.sidebar.success("ML predictions loaded")
            else:
                ml_predictions = pd.DataFrame()
                
        except Exception as e:
            st.sidebar.warning(f"ML predictions not available: {str(e)}")
            ml_predictions = pd.DataFrame()
        
        return predictions, standings, players, ml_predictions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def predict_player_performance(player, opponent_team, standings):
    """Predict single player's goal/assist/point probability"""
    
    # Base rates
    goals_pg = player['goals_pg']
    assists_pg = player['assists_pg']
    
    # Get opponent defense strength
    opponent = standings[standings['Team'] == opponent_team].iloc[0]
    
    # Calculate total goals against (home + away average)
    standings['Total Goals Against'] = (standings['Home Goals Against'] + standings['Away Goals Against']) / 2
    league_avg_ga = standings['Total Goals Against'].mean()
    defensive_factor = opponent['Total Goals Against'] / league_avg_ga
    
    # Shot volume matters
    shot_factor = min(player['shots_pg'] / 2.5, 1.5) if player['shots_pg'] > 0 else 1.0
    
    # Adjust probabilities
    goal_prob = goals_pg * defensive_factor * shot_factor
    goal_prob = min(0.90, max(0.01, goal_prob))
    
    assist_prob = assists_pg * defensive_factor
    assist_prob = min(0.90, max(0.01, assist_prob))
    
    # Probability of getting at least one point
    point_prob = 1 - ((1 - goal_prob) * (1 - assist_prob))
    
    return {
        'player': player['Player'],
        'position': player.get('Pos', 'F'),
        'goal_prob': goal_prob,
        'assist_prob': assist_prob,
        'point_prob': point_prob
    }

def get_top_players_for_game(home_team, away_team, players, standings, top_n=5):
    """Get top predicted players for a specific game"""
    
    # Get team abbreviations
    home_abbrev = TEAM_TO_ABBREV.get(home_team)
    away_abbrev = TEAM_TO_ABBREV.get(away_team)
    
    if not home_abbrev or not away_abbrev:
        return None
    
    # Get players from both teams
    home_players = players[players['Team'] == home_abbrev].copy()
    away_players = players[players['Team'] == away_abbrev].copy()
    
    all_predictions = []
    
    # Predict for each player
    for _, player in home_players.iterrows():
        pred = predict_player_performance(player, away_team, standings)
        pred['team'] = home_team
        pred['team_abbrev'] = home_abbrev
        all_predictions.append(pred)
    
    for _, player in away_players.iterrows():
        pred = predict_player_performance(player, home_team, standings)
        pred['team'] = away_team
        pred['team_abbrev'] = away_abbrev
        all_predictions.append(pred)
    
    # Convert to DataFrame
    preds_df = pd.DataFrame(all_predictions)
    
    # Get top scorers
    top_goals = preds_df.nlargest(top_n, 'goal_prob')
    top_assists = preds_df.nlargest(top_n, 'assist_prob')
    top_points = preds_df.nlargest(top_n, 'point_prob')
    
    return {
        'top_goals': top_goals.to_dict('records'),
        'top_assists': top_assists.to_dict('records'),
        'top_points': top_points.to_dict('records')
    }

def calculate_excel_prediction(home_team, away_team, standings, predictions, game_date):
    """Calculate Excel model prediction"""
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
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
    
    if predicted_winner is None:
        home_home_win_pct = home_row['HomeWin%']
        away_away_win_pct = away_row['AwayWin%']
        homeice_diff = (home_home_win_pct - away_away_win_pct) * 6
        predicted_winner = home_team if homeice_diff > 0 else away_team
    
    home_goals_for = home_row['Home Goals per Game']
    home_goals_against = home_row['Home Goals Against']
    away_goals_for = away_row['Away Goals per Game']
    away_goals_against = away_row['Away Goals Against']
    
    expected_home = (home_goals_for + away_goals_against) / 2
    expected_away = (away_goals_for + home_goals_against) / 2
    
    home_adjustment = homeice_diff * 0.5
    predicted_home = max(2, min(7, round(expected_home + home_adjustment)))
    predicted_away = max(2, min(7, round(expected_away - home_adjustment)))
    
    if predicted_winner == home_team and predicted_home <= predicted_away:
        predicted_home = predicted_away + 1
    elif predicted_winner == away_team and predicted_away <= predicted_home:
        predicted_away = predicted_home + 1
    
    win_prob = 0.5 + (abs(homeice_diff) / 12)
    win_prob = min(0.85, max(0.52, win_prob))
    
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
    
    try:
        ml_home_score = int(float(ml_game.get('ml_predicted_home_score', 3)))
        ml_away_score = int(float(ml_game.get('ml_predicted_away_score', 2)))
    except:
        ml_home_score = 3
        ml_away_score = 2
    
    ml_is_overtime = ml_game.get('ml_is_overtime', False) if 'ml_is_overtime' in ml_game.index else False
    ml_confidence = ml_game.get('ml_confidence', 0.5)
    
    return {
        'winner': ml_game['ml_predicted_winner'],
        'home_score': ml_home_score,
        'away_score': ml_away_score,
        'confidence': ml_confidence,
        'is_overtime': ml_is_overtime
    }

def display_game(game, standings, predictions, players, ml_predictions):
    """Display a single game prediction with player predictions"""
    home_team = game['Home']
    away_team = game['Visitor']
    game_time = game['Time']
    game_date = game['Date']
    
    # Get predictions
    excel_pred = calculate_excel_prediction(home_team, away_team, standings, predictions, game_date)
    ml_pred = get_ml_prediction(home_team, away_team, game_date, ml_predictions)
    player_preds = get_top_players_for_game(home_team, away_team, players, standings, top_n=5)
    
    # Get team records
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Header
        st.markdown(f"**{game_time}**")
        
        # Teams
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {away_team}")
            st.caption(f"Record: {int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])} | {int(away_row['PTS'])} pts")
        with col2:
            st.markdown(f"### {home_team}")
            st.caption(f"Record: {int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['OTL'])} | {int(home_row['PTS'])} pts")
        
        st.divider()
        
        # Game Predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("**Excel Model**")
            st.markdown(f'<div class="winner-text">Winner: {excel_pred["winner"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-text">Score: {excel_pred["away_score"]}-{excel_pred["home_score"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Confidence: {excel_pred["confidence"]:.1%}</div>', unsafe_allow_html=True)
            if excel_pred['ot_probability'] > 0.25:
                st.markdown(f'<div class="metric-label">OT Probability: {excel_pred["ot_probability"]:.1%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("**ML Model**")
            if ml_pred:
                st.markdown(f'<div class="winner-text">Winner: {ml_pred["winner"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-text">Score: {ml_pred["away_score"]}-{ml_pred["home_score"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Confidence: {ml_pred["confidence"]:.1%}</div>', unsafe_allow_html=True)
                if ml_pred['is_overtime']:
                    st.markdown('<div class="metric-label">Overtime Predicted</div>', unsafe_allow_html=True)
            else:
                st.info("No ML prediction available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Agreement
        if ml_pred and excel_pred['winner'] == ml_pred['winner']:
            st.success("Both models agree on winner")
        elif ml_pred:
            st.warning("Models predict different winners")
        
        # Player Predictions
        if player_preds:
            st.divider()
            st.markdown("**Top Player Predictions**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Top Goal Scorers**")
                for i, player in enumerate(player_preds['top_goals'][:5], 1):
                    st.markdown(f"{i}. **{player['player']}** ({player['team_abbrev']})")
                    st.caption(f"{player['goal_prob']:.1%} chance")
            
            with col2:
                st.markdown("**Top Assist Leaders**")
                for i, player in enumerate(player_preds['top_assists'][:5], 1):
                    st.markdown(f"{i}. **{player['player']}** ({player['team_abbrev']})")
                    st.caption(f"{player['assist_prob']:.1%} chance")
            
            with col3:
                st.markdown("**Top Point Getters**")
                for i, player in enumerate(player_preds['top_points'][:5], 1):
                    st.markdown(f"{i}. **{player['player']}** ({player['team_abbrev']})")
                    st.caption(f"{player['point_prob']:.1%} chance")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    predictions, standings, players, ml_predictions = load_data()
    
    if predictions is None or standings is None or players is None:
        return
    
    st.title("NHL Prediction Model 2025-26")
    eastern_now = datetime.now(EASTERN)
    st.caption(f"Last updated: {eastern_now.strftime('%Y-%m-%d %I:%M:%S %p')} ET")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    page = st.sidebar.radio("Select Page", ["Today's Games", "Custom Matchup", "Performance"])
    
    if page == "Today's Games":
        eastern_now = datetime.now(EASTERN)
        today = pd.Timestamp(eastern_now.date()).normalize()
        todays_games = predictions[predictions['Date'] == today].copy()
        
        if len(todays_games) == 0:
            st.warning("No games scheduled today")
            
            future_games = predictions[predictions['Date'] > today].head(5)
            if len(future_games) > 0:
                st.subheader("Upcoming Games")
                for _, game in future_games.iterrows():
                    st.info(f"{game['Visitor']} @ {game['Home']} - {game['Date'].strftime('%B %d, %Y')}")
        else:
            st.subheader(f"Today's Games ({len(todays_games)})")
            todays_games = todays_games.sort_values('Time')
            for _, game in todays_games.iterrows():
                display_game(game, standings, predictions, players, ml_predictions)
    
    elif page == "Custom Matchup":
        st.subheader("Custom Matchup")
        
        teams = sorted(standings['Team'].tolist())
        
        col1, col2 = st.columns(2)
        with col1:
            away_team = st.selectbox("Away Team", ["Select..."] + teams)
        with col2:
            home_team = st.selectbox("Home Team", ["Select..."] + teams)
        
        if st.button("Generate Prediction"):
            if away_team != "Select..." and home_team != "Select..." and away_team != home_team:
                eastern_now = datetime.now(EASTERN)
                today = pd.Timestamp(eastern_now.date()).normalize()
                
                excel_pred = calculate_excel_prediction(home_team, away_team, standings, None, None)
                ml_pred = get_ml_prediction(home_team, away_team, today, ml_predictions)
                player_preds = get_top_players_for_game(home_team, away_team, players, standings, top_n=5)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Excel Model**")
                    st.write(f"Winner: {excel_pred['winner']}")
                    st.write(f"Score: {excel_pred['away_score']}-{excel_pred['home_score']}")
                    st.write(f"Confidence: {excel_pred['confidence']:.1%}")
                
                with col2:
                    st.markdown("**ML Model**")
                    if ml_pred:
                        st.write(f"Winner: {ml_pred['winner']}")
                        st.write(f"Score: {ml_pred['away_score']}-{ml_pred['home_score']}")
                        st.write(f"Confidence: {ml_pred['confidence']:.1%}")
                    else:
                        st.info("No ML prediction")
                
                # Player predictions
                if player_preds:
                    st.divider()
                    st.markdown("**Top Player Predictions**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Top Goal Scorers**")
                        for i, player in enumerate(player_preds['top_goals'][:5], 1):
                            st.write(f"{i}. {player['player']} ({player['team_abbrev']}) - {player['goal_prob']:.1%}")
                    
                    with col2:
                        st.markdown("**Top Assist Leaders**")
                        for i, player in enumerate(player_preds['top_assists'][:5], 1):
                            st.write(f"{i}. {player['player']} ({player['team_abbrev']}) - {player['assist_prob']:.1%}")
                    
                    with col3:
                        st.markdown("**Top Point Getters**")
                        for i, player in enumerate(player_preds['top_points'][:5], 1):
                            st.write(f"{i}. {player['player']} ({player['team_abbrev']}) - {player['point_prob']:.1%}")
            else:
                st.error("Please select two different teams")
    
    elif page == "Performance":
        st.subheader("Model Performance")
        
        # Excel performance
        st.markdown("**Excel Model**")
        completed = predictions[predictions['Locked Correct'].isin(['YES', 'NO'])].copy()
        if len(completed) > 0:
            total = len(completed)
            correct = (completed['Locked Correct'] == 'YES').sum()
            accuracy = (correct / total * 100)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Games", total)
            col2.metric("Correct", correct)
            col3.metric("Accuracy", f"{accuracy:.1f}%")
        else:
            st.info("No completed games")
        
        st.divider()
        
        # ML performance
        st.markdown("**ML Model**")
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
                    ml_accuracy = (ml_correct / ml_total * 100)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Games", ml_total)
                    col2.metric("Correct", int(ml_correct))
                    col3.metric("Accuracy", f"{ml_accuracy:.1f}%")
                else:
                    st.info("No completed games")
            else:
                st.info("No completed games")
        else:
            st.info("ML model not available")

if __name__ == "__main__":
    main()
