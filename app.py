from flask import Flask, render_template, request, jsonify, redirect, url_for
from apscheduler.schedulers.background import BackgroundScheduler
import joblib, numpy as np, pandas as pd, json, os, logging, requests
from datetime import datetime
from fuzzywuzzy import process

app = Flask(__name__)
API_KEY = 'f22cd4651da7f18126549e07566740c3'

# Load models
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

# --- Utils ---
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
    return round(min(kelly_fraction * base_unit, max_units), 2)

def sigmoid_from_line(predicted, line, scaling=1.5):
    return np.clip(1 / (1 + np.exp(-scaling * (predicted - line))), 0.01, 0.99)

def format_output(date, game, bet_type, bookmaker, odds, implied, model_prob, ev, units,
                  money_line_bet=None, total_bet=None, hits_bet=None, strikeouts_bet=None,
                  pitcher_props_bet=None, predicted_total=None):
    return {
        'Date': date.strftime('%Y-%m-%d'),
        'Game': game,
        'Bet Type': bet_type,
        'Bookmaker': bookmaker,
        'Odds': odds,
        'Implied Prob': f"{implied:.2%}",
        'Model Prob': f"{model_prob:.2%}",
        'EV %': f"{ev:.2%}",
        'Units to Wager': f"{units:.2f}",
        'Money Line Bet': money_line_bet or '',
        'Total Bet': total_bet or '',
        'Hits Bet': hits_bet or '',
        'Strikeouts Bet': strikeouts_bet or '',
        'Pitcher Props Bet': pitcher_props_bet or '',
        'Predicted Total': f"{predicted_total:.2f}" if predicted_total else ''
    }

def save_to_json(data, filename='output.json'):
    os.makedirs('mlb_project', exist_ok=True)
    with open(f'mlb_project/{filename}', 'w') as f:
        json.dump(data, f, indent=4)

def load_results():
    try:
        with open('mlb_project/output.json', 'r') as f:
            return json.load(f)
    except:
        return []

# --- Predictions ---
def predict_moneyline(home, away):
    h = le_team.transform([home])[0]
    a = le_team.transform([away])[0]
    return np.clip(moneyline_model.predict_proba([[h, a]])[0][1], 0.01, 0.99)

def predict_total(home, away):
    h = le_team.transform([home])[0]
    a = le_team.transform([away])[0]
    total = total_runs_model.predict(pd.DataFrame([{'home_team_encoded': h, 'away_team_encoded': a}]))[0]
    return total, sigmoid_from_line(total, 8.5)

def predict_hits(team, opponent, venue, ip=1, bb=0, er=0, h=0, r=0):
    x = [[le_opponent.transform([opponent])[0], le_venue.transform([venue])[0], ip, bb, er, h, r]]
    return np.clip(hits_model.predict(x)[0] / 10, 0.01, 0.99)

def predict_strikeouts(team, opponent, venue, ip=1, bb=0, er=0, h=0, r=0):
    x = [[le_team.transform([team])[0], le_opponent.transform([opponent])[0], le_venue.transform([venue])[0], ip, bb, er, h, r]]
    return np.clip(strikeouts_model.predict(x)[0] / 10, 0.01, 0.99)

def predict_pitcher_props(opponent, ip=1, avg_so=1.5, games=30):
    x = [[ip, avg_so, games, le_opponent.transform([opponent])[0]]]
    return np.clip(pitcher_props_model.predict(x)[0] / 10, 0.01, 0.99)

def get_odds():
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    try:
        res = requests.get(url, params={
            'apiKey': API_KEY, 'regions': 'us',
            'bookmakers': 'fanduel,draftkings', 'markets': 'h2h,totals',
            'oddsFormat': 'american'
        })
        return res.json()
    except Exception as e:
        logger.error(f"API Error: {e}")
        return []

# --- Main Prediction Logic ---
def run_daily_predictions():
    odds_data = get_odds()
    results, seen = [], set()

    if odds_data:
        for game in odds_data:
            try:
                home = smart_team_match(game['home_team'])
                away = smart_team_match(game['away_team'])

                game_df = pd.read_csv('mlb_data/mlb_games.csv')
                row = game_df[(game_df['home_team_name'] == home) & (game_df['away_team_name'] == away)]
                if row.empty: continue

                venue = row.iloc[0].get('venue', 'Unknown')
                pitcher = full_pitching_df[full_pitching_df['home_team_name'] == home]
                pitcher_name = pitcher.iloc[0]['player_name'] if not pitcher.empty else 'Unknown'

                moneyline = predict_moneyline(home, away)
                total_raw, total_sigmoid = predict_total(home, away)
                hits_prob = predict_hits(home, away, venue)
                strikeouts_prob = predict_strikeouts(home, away, venue)
                pitcher_props_prob = predict_pitcher_props(away)

                for bookmaker in game['bookmakers']:
                    bm = bookmaker['title']
                    seen_points = set()
                    added_ml = False

                    for market in bookmaker['markets']:
                        for outcome in market['outcomes']:
                            odds = int(outcome.get('price', 0))
                            implied = convert_odds_to_implied_prob(odds)
                            key = (game['id'], market['key'], bm, outcome.get('name'), outcome.get('point', ''))

                            if key in seen: continue
                            seen.add(key)

                            if market['key'] == 'h2h' and not added_ml:
                                model_prob = moneyline if outcome['name'] == home else 1 - moneyline
                                ev = calculate_ev(model_prob, implied, odds)
                                units = suggest_wager(model_prob, odds)
                                results.append(format_output(datetime.now(), f"{home} vs {away}", "Moneyline", bm, odds, implied, model_prob, ev, units, money_line_bet=home if moneyline > 0.5 else away))
                                added_ml = True

                            elif market['key'] == 'totals':
                                point = outcome.get('point')
                                if point in seen_points: continue
                                seen_points.add(point)
                                ev = calculate_ev(total_sigmoid, implied, odds)
                                units = suggest_wager(total_sigmoid, odds)
                                results.append(format_output(datetime.now(), f"{home} vs {away}", f"Total {point}", bm, odds, implied, total_sigmoid, ev, units, total_bet="OVER" if total_sigmoid > 0.5 else "UNDER", predicted_total=total_raw))

                # Add model-only props
                results.append(format_output(datetime.now(), f"{home} vs {away}", "Hits Bet", "Model", 100, 0.5, hits_prob, calculate_ev(hits_prob, 0.5, 100), suggest_wager(hits_prob, 100), hits_bet="OVER" if hits_prob > 0.5 else "UNDER"))
                results.append(format_output(datetime.now(), f"{home} vs {away}", "Strikeouts Bet", "Model", 100, 0.5, strikeouts_prob, calculate_ev(strikeouts_prob, 0.5, 100), suggest_wager(strikeouts_prob, 100), strikeouts_bet="OVER" if strikeouts_prob > 0.5 else "UNDER"))
                results.append(format_output(datetime.now(), f"{home} vs {away}", "Pitcher Props Bet", "Model", 100, 0.5, pitcher_props_prob, calculate_ev(pitcher_props_prob, 0.5, 100), suggest_wager(pitcher_props_prob, 100), pitcher_props_bet=pitcher_name))
            except Exception as e:
                logger.error(f"Game error: {e}")

    save_to_json(results)
    return results

# --- Routes ---
@app.route('/')
def index():
    data = load_results()
    bookmaker = request.args.get('bookmaker')
    bet_type = request.args.get('bet_type')
    if bookmaker: data = [r for r in data if r['Bookmaker'].lower() == bookmaker.lower()]
    if bet_type: data = [r for r in data if r['Bet Type'].lower().startswith(bet_type.lower())]
    return render_template("index.html", results=data, bookmakers=['fanduel', 'draftkings', 'model'],
                           bet_types=['moneyline', 'total', 'hits', 'strikeouts', 'pitcher props'])

@app.route('/run-now', methods=['POST'])
def run_now():
    run_daily_predictions()
    return redirect(url_for('index'))

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(str(e))
    return jsonify(error=str(e)), 500

# Schedule background job
scheduler = BackgroundScheduler()
scheduler.add_job(run_daily_predictions, 'interval', hours=24)
scheduler.start()

if __name__ == '__main__':
    run_daily_predictions()
    app.run(debug=True)
