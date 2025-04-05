from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import joblib
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import requests
import logging
from fuzzywuzzy import process

app = Flask(__name__)
API_KEY = 'f22cd4651da7f18126549e07566740c3'

# Load models and encoders
hits_model = joblib.load("models/hits_model.pkl")
moneyline_model = joblib.load("models/moneyline_model.pkl")
pitcher_props_model = joblib.load("models/pitcher_props_model.pkl")
strikeouts_model = joblib.load("models/strikeouts_model.pkl")
total_runs_model = joblib.load("models/total_runs_model.pkl")
le_team = joblib.load("models/team_encoder.pkl")
le_venue = joblib.load("models/venue_encoder.pkl")
le_opponent = joblib.load("models/opponent_encoder.pkl")
MODEL_TEAM_NAMES = le_team.classes_.tolist()

full_pitching_df = pd.read_csv("mlb_data/full_pitching_dataset.csv")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def smart_team_match(api_team):
    match, score = process.extractOne(api_team, MODEL_TEAM_NAMES)
    return match if score >= 80 else api_team

def convert_odds_to_implied_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)

def calculate_ev(model_prob, implied_prob, odds):
    decimal_odds = (odds / 100) + 1 if odds > 0 else (100 / -odds) + 1
    return (model_prob * decimal_odds) - (1 - model_prob)

def suggest_wager(model_prob, odds, base_unit=1.0, max_units=3.0):
    if odds == 0:
        return 0.0
    decimal_odds = (odds / 100) + 1 if odds > 0 else (100 / -odds) + 1
    b = decimal_odds - 1
    q = 1 - model_prob
    kelly_fraction = ((b * model_prob) - q) / b
    kelly_fraction = max(0, kelly_fraction)
    units = kelly_fraction * base_unit
    return round(min(units, max_units), 2)

def sigmoid_from_line(predicted_value, betting_line, scaling_factor=1.5):
    diff = predicted_value - betting_line
    return np.clip(1 / (1 + np.exp(-scaling_factor * diff)), 0.01, 0.99)

def format_output(date, game, bet_type, bookmaker, odds, implied_prob, model_prob, ev, units,
                  money_line_bet=None, total_bet=None, hits_bet=None, strikeouts_bet=None, pitcher_props_bet=None):
    return {
        'Date': date.strftime('%Y-%m-%d'),
        'Game': game,
        'Bet Type': bet_type,
        'Bookmaker': bookmaker,
        'Odds': odds,
        'Implied Prob': f"{implied_prob:.2%}",
        'Model Prob': f"{model_prob:.2%}",
        'EV %': f"{ev:.2%}",
        'Units to Wager': f"{units:.2f}",
        'Money Line Bet': money_line_bet or '',
        'Total Bet': total_bet or '',
        'Hits Bet': hits_bet or '',
        'Strikeouts Bet': strikeouts_bet or '',
        'Pitcher Props Bet': pitcher_props_bet or ''
    }

def save_to_json(data, filename='output.json'):
    os.makedirs('mlb_project', exist_ok=True)
    with open(os.path.join('mlb_project', filename), 'w') as f:
        json.dump(data, f, indent=4)

def save_top_ev_bets(results, top_n=3, filename='top_bets.json'):
    sorted_results = sorted(
        [r for r in results if float(r['EV %'].strip('%')) > 0],
        key=lambda x: float(x['EV %'].strip('%')),
        reverse=True
    )
    top_bets = sorted_results[:top_n]
    with open(os.path.join('mlb_project', filename), 'w') as f:
        json.dump(top_bets, f, indent=4)

def load_results():
    try:
        with open('mlb_project/output.json', 'r') as f:
            return json.load(f)
    except:
        return []

# Prediction models
def predict_moneyline(home_team, away_team):
    home_encoded = le_team.transform([home_team])[0]
    away_encoded = le_team.transform([away_team])[0]
    prob = moneyline_model.predict_proba([[home_encoded, away_encoded]])[0][1]
    return np.clip(prob, 0.01, 0.99)

def predict_total_runs(home_team, away_team):
    df = pd.DataFrame([{
        'home_team_encoded': le_team.transform([home_team])[0],
        'away_team_encoded': le_team.transform([away_team])[0]
    }])
    total = total_runs_model.predict(df)[0]
    return sigmoid_from_line(total, 8.5)

def predict_hits(team, opponent, venue, ip, bb, er, h, r):
    input_data = np.array([[le_opponent.transform([opponent])[0],
                            le_venue.transform([venue])[0], ip, bb, er, h, r]])
    predicted_hits = hits_model.predict(input_data)[0]
    return np.clip(predicted_hits / 10.0, 0.01, 0.99)

def predict_strikeouts(team, opponent, venue, ip, bb, er, h, r):
    input_data = np.array([[le_team.transform([team])[0],
                            le_opponent.transform([opponent])[0],
                            le_venue.transform([venue])[0], ip, bb, er, h, r]])
    predicted_so = strikeouts_model.predict(input_data)[0]
    return np.clip(predicted_so / 10.0, 0.01, 0.99)

def predict_pitcher_props(opponent, ip, rolling_so_avg, career_games):
    input_data = np.array([[ip, rolling_so_avg, career_games,
                            le_opponent.transform([opponent])[0]]])
    predicted = pitcher_props_model.predict(input_data)[0]
    return np.clip(predicted / 10.0, 0.01, 0.99)

def get_odds():
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    try:
        response = requests.get(url, params={
            'apiKey': API_KEY,
            'regions': 'us',
            'bookmakers': 'fanduel,draftkings',
            'markets': 'h2h,totals',
            'oddsFormat': 'american'
        })
        return response.json()
    except Exception as e:
        logger.error(f"API Error: {e}")
        return []

def run_daily_predictions():
    odds_data = get_odds()
    results = []
    seen_keys = set()

    if odds_data:
        for game in odds_data:
            try:
                home_team = smart_team_match(game.get('home_team', ''))
                away_team = smart_team_match(game.get('away_team', ''))
                game_df = pd.read_csv('mlb_data/mlb_games.csv')
                row = game_df[(game_df['home_team_name'] == home_team) & (game_df['away_team_name'] == away_team)]
                if row.empty:
                    continue
                game_row = row.iloc[0]
                venue = game_row.get('venue', 'Unknown')

                pitcher_row = full_pitching_df[(full_pitching_df['home_team_name'] == home_team)]
                pitcher_name = pitcher_row.iloc[0]['player_name'] if not pitcher_row.empty else 'Unknown'
                pitcher_props_prob = predict_pitcher_props(away_team, 1.0, 1.5, 30)
                moneyline = predict_moneyline(home_team, away_team)
                total = predict_total_runs(home_team, away_team)
                hits_prob = predict_hits(home_team, away_team, venue, 1.0, 0, 0, 0, 0)
                strikeouts_prob = predict_strikeouts(home_team, away_team, venue, 1.0, 0, 0, 0, 0)

                for bookmaker in game.get('bookmakers', []):
                    bookmaker_name = bookmaker.get('title', '')
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key')
                        for outcome in market.get('outcomes', []):
                            try:
                                odds = int(outcome.get('price', 0))
                                implied = convert_odds_to_implied_prob(odds)
                                key = (game['id'], market_key, bookmaker_name, outcome.get('name'), outcome.get('point', ''))
                                if key in seen_keys:
                                    continue
                                seen_keys.add(key)

                                if market_key == 'h2h':
                                    model_prob = moneyline if outcome['name'] == home_team else 1 - moneyline
                                    ev = calculate_ev(model_prob, implied, odds)
                                    units = suggest_wager(model_prob, odds)
                                    results.append(format_output(datetime.now(), f"{home_team} vs {away_team}", "Moneyline",
                                                                 bookmaker_name, odds, implied, model_prob, ev, units,
                                                                 money_line_bet=home_team if moneyline > 0.5 else away_team))
                                elif market_key == 'totals':
                                    point = outcome.get('point')
                                    model_prob = total
                                    ev = calculate_ev(model_prob, implied, odds)
                                    units = suggest_wager(model_prob, odds)
                                    results.append(format_output(datetime.now(), f"{home_team} vs {away_team}", f"Total {point}",
                                                                 bookmaker_name, odds, implied, model_prob, ev, units,
                                                                 total_bet="OVER" if total > 0.5 else "UNDER"))
                            except Exception as inner_err:
                                logger.error(f"Error processing outcome: {inner_err}")

                # Add model predictions (no bookmaker odds for these yet)
                results.append(format_output(datetime.now(), f"{home_team} vs {away_team}", "Hits Bet", "Model", 100, 0.5, hits_prob,
                                             calculate_ev(hits_prob, 0.5, 100), suggest_wager(hits_prob, 100),
                                             hits_bet="OVER" if hits_prob > 0.5 else "UNDER"))
                results.append(format_output(datetime.now(), f"{home_team} vs {away_team}", "Strikeouts Bet", "Model", 100, 0.5, strikeouts_prob,
                                             calculate_ev(strikeouts_prob, 0.5, 100), suggest_wager(strikeouts_prob, 100),
                                             strikeouts_bet="OVER" if strikeouts_prob > 0.5 else "UNDER"))
                results.append(format_output(datetime.now(), f"{home_team} vs {away_team}", "Pitcher Props Bet", "Model", 100, 0.5, pitcher_props_prob,
                                             calculate_ev(pitcher_props_prob, 0.5, 100), suggest_wager(pitcher_props_prob, 100),
                                             pitcher_props_bet=pitcher_name))

            except Exception as e:
                logger.error(f"Error processing game: {e}")

    save_to_json(results)
    save_top_ev_bets(results)

@app.route('/')
def index():
    data = load_results()
    bookmaker = request.args.get('bookmaker')
    bet_type = request.args.get('bet_type')
    if bookmaker:
        data = [r for r in data if r['Bookmaker'].lower() == bookmaker.lower()]
    if bet_type:
        data = [r for r in data if r['Bet Type'].lower().startswith(bet_type.lower())]
    return render_template('index.html', results=data, bookmakers=['fanduel', 'draftkings', 'model'],
                           bet_types=['moneyline', 'total', 'hits', 'strikeouts', 'pitcher props'])

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(str(e))
    return jsonify(error=str(e)), 500

# Schedule
scheduler = BackgroundScheduler()
scheduler.add_job(run_daily_predictions, 'interval', hours=24)
scheduler.start()

if __name__ == '__main__':
    run_daily_predictions()
    app.run(debug=True)
