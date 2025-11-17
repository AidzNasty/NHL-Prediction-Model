#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:59:42 2025

@author: aidanconte
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHL Prediction Model - Web App Version
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

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .stApp {
        background-color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    .game-card {
        background-color: #16213e;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00d4ff;
        margin: 10px 0px;
    }
    .winner {
        color: #4caf50;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #16213e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #533483;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
EXCEL_FILE = 'Aidan Conte NHL 2025-26 Prediction Model.xlsx'
LEAGUE_AVG_TOTAL = 6.24
TEAM_WEIGHT = 0.70
HOMEICE_WEIGHT = 0.30

@st.cache_data
def load_data():
    """Load Excel data with caching"""
    try:
        predictions = pd.read_excel(EXCEL_FILE, sheet_name='NHL HomeIce Model', header=0)
        predictions = predictions.iloc[:, 3:].reset_index(drop=True)
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        return predictions, standings
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {e}")
        st.info("Make sure 'Aidan Conte NHL 2025-26 Prediction Model.xlsx' is in the same folder as this script")
        return None, None

def calculate_prediction(home_team, away_team, standings):
    """Calculate prediction for a matchup"""
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    # Calculate HomeIce Differential
    home_home_win_pct = home_row['HomeWin%']
    away_away_win_pct = away_row['AwayWin%']
    homeice_diff = (home_home_win_pct - away_away_win_pct) * 6

    #homeice_diff = extract_excel_data('Aidan Conte NHL 2025-26 Prediction Model.xlsx', 'NHL HomeIce Model', 'HomeIce Differential')
    
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

def display_game_card(game, standings):
    """Display a game card with prediction"""
    home_team = game['Home']
    away_team = game['Visitor']
    game_time = game['Time']
    
    prediction = calculate_prediction(home_team, away_team, standings)
    
    home_row = prediction['home_row']
    away_row = prediction['away_row']
    
    # Create card
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Time
        st.markdown(f"🕐 **{game_time}**")
        
        # Teams
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"### {away_team}")
            st.caption(f"{int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])} ({int(away_row['PTS'])} pts)")
        
        with col2:
            st.markdown("### @")
        
        with col3:
            st.markdown(f"### {home_team}")
            st.caption(f"{int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['OTL'])} ({int(home_row['PTS'])} pts)")
        
        st.markdown("---")
        
        # Prediction
        st.markdown(f'<div class="winner">🏆 {prediction["predicted_winner"]}</div>', unsafe_allow_html=True)
        st.markdown(f"**Win Probability:** {prediction['win_prob']:.1%}")
        st.markdown(f"**Predicted Score:** {away_team} {prediction['predicted_away']} - {prediction['predicted_home']} {home_team}")
        st.caption(f"HomeIce Differential: {prediction['homeice_diff']:+.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Load data
    predictions, standings = load_data()
    
    if predictions is None or standings is None:
        return
    
    # Title
    st.title("🏒 NHL Prediction Model 2025-26")
    st.markdown(f"### 📅 {datetime.now().strftime('%A, %B %d, %Y')}")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Today's Games", "Custom Matchup", "Model Performance"])
    
    # TODAY'S GAMES PAGE
    if page == "Today's Games":
        st.markdown("---")
        
        # Get today's games
        today = datetime.now().date()
        todays_games = predictions[predictions['Date'].dt.date == today].copy()
        
        if len(todays_games) == 0:
            st.warning("⚠️ No games scheduled for today")
            
            # Show upcoming games
            future_games = predictions[predictions['Date'].dt.date > today].copy()
            if len(future_games) > 0:
                st.subheader("📅 Next Upcoming Games:")
                future_games = future_games.sort_values('Date').head(5)
                
                for idx, game in future_games.iterrows():
                    date_str = game['Date'].strftime('%A, %B %d')
                    st.info(f"{game['Visitor']} @ {game['Home']} - {date_str}")
        else:
            st.subheader(f"🏒 Today's Games ({len(todays_games)} matchups)")
            
            # Sort by time
            todays_games = todays_games.sort_values('Time')
            
            # Display each game
            for idx, game in todays_games.iterrows():
                display_game_card(game, standings)
    
    # CUSTOM MATCHUP PAGE
    elif page == "Custom Matchup":
        st.markdown("---")
        st.subheader("⚙️ Custom Matchup Prediction")
        
        teams = sorted(standings['Team'].tolist())
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            away_team = st.selectbox("Away Team", [""] + teams, key="away")
        
        with col2:
            st.markdown("### @")
        
        with col3:
            home_team = st.selectbox("Home Team", [""] + teams, key="home")
        
        if st.button("🎯 Generate Prediction", type="primary"):
            if not away_team or not home_team:
                st.warning("⚠️ Please select both teams")
            elif away_team == home_team:
                st.error("❌ Please select different teams")
            else:
                st.markdown("---")
                prediction = calculate_prediction(home_team, away_team, standings)
                
                # Display prediction
                st.markdown('<div class="game-card">', unsafe_allow_html=True)
                
                home_row = prediction['home_row']
                away_row = prediction['away_row']
                
                # Teams
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.markdown(f"### {away_team}")
                    st.caption(f"{int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['OTL'])} ({int(away_row['PTS'])} pts)")
                
                with col2:
                    st.markdown("### @")
                
                with col3:
                    st.markdown(f"### {home_team}")
                    st.caption(f"{int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['OTL'])} ({int(home_row['PTS'])} pts)")
                
                st.markdown("---")
                
                # Prediction
                st.markdown(f'<div class="winner">🏆 {prediction["predicted_winner"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**Win Probability:** {prediction['win_prob']:.1%}")
                st.markdown(f"**Predicted Score:** {away_team} {prediction['predicted_away']} - {prediction['predicted_home']} {home_team}")
                st.caption(f"HomeIce Differential: {prediction['homeice_diff']:+.3f}")
                
                # Team stats comparison
                st.markdown("### Team Statistics")
                
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
                
                st.table(stats_data)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # MODEL PERFORMANCE PAGE
    elif page == "Model Performance":
        st.markdown("---")
        st.subheader("📊 Model Performance Statistics")
        
        correct_col = 'Locked Correct'
        completed = predictions[predictions[correct_col].isin(['YES', 'NO'])].copy()
        
        if len(completed) == 0:
            st.info("No completed games yet")
        else:
            completed = completed.sort_values('Date')
            
            # Overall stats
            total_games = len(completed)
            correct_games = (completed[correct_col] == 'YES').sum()
            overall_accuracy = (correct_games / total_games * 100)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📈 Overall Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
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
                st.markdown("---")
                last_20 = past_games.tail(20)
                correct_20 = (last_20[correct_col] == 'YES').sum()
                accuracy_20 = (correct_20 / 20 * 100)
                
                first_date = last_20['Date'].min().strftime('%Y-%m-%d')
                last_date = last_20['Date'].max().strftime('%Y-%m-%d')
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Last 20 Games")
                st.caption(f"{first_date} to {last_date}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Correct", f"{correct_20}/20")
                
                with col2:
                    st.metric("Accuracy", f"{accuracy_20:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent games breakdown
            st.markdown("---")
            st.markdown("### 📋 Recent Games Breakdown")
            
            recent_games = past_games.tail(10)
            
            display_data = []
            for idx, game in recent_games.iterrows():
                result = "✅" if game[correct_col] == 'YES' else "❌"
                display_data.append({
                    "Date": game['Date'].strftime('%Y-%m-%d'),
                    "Matchup": f"{game['Visitor']} @ {game['Home']}",
                    "Prediction": game['Locked Prediction'],
                    "Result": result
                })
            
            st.table(display_data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **NHL Prediction Model 2025-26**
    
    Using HomeIce Differential and team statistics to predict game outcomes.
    
    Model blends:
    - 70% team-based stats
    - 30% HomeIce advantage
    """)

if __name__ == "__main__":
    main()
