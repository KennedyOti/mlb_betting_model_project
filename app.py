from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_caching import Cache
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import joblib, numpy as np, pandas as pd, json, os, logging, requests, time
from datetime import datetime, timedelta
from fuzzywuzzy import process
import xgboost, sklearn
import pytz
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
API_KEY = '660d6e7d9e30433ab657a4661f7b8520'

# Define Eastern Time timezone
eastern_tz = pytz.timezone('America/New_York')

# Configure Flask-Caching
cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300  # 5 minutes
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# Setup logging with Eastern Time
class EasternTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=eastern_tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = EasternTimeFormatter('%(asctime)s %(levelname)s %(message)s')
file_handler = RotatingFileHandler('mlb_app.log', maxBytes=1000000, backupCount=5)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.handlers = [file_handler, stream_handler]
logger.info(f"XGBoost version: {xgboost.__version__}")
logger.info(f"scikit-learn version: {sklearn.__version__}")

# Load models
try:
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
    logger.info("All models and data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or data: {e}")
    raise

# --- Utils ---
def smart_team_match(api_team):
    match, score = process.extractOne(api_team, MODEL_TEAM_NAMES)
    if score < 80:
        logger.warning(f"Low match score for '{api_team}': matched '{match}' with score {score}")
        # Manual mapping for known discrepancies
        manual_mapping = {
            "NY Yankees": "New York Yankees",
            "Chi White Sox": "Chicago White Sox",
            "Chi Cubs": "Chicago Cubs",
            "LA Dodgers": "Los Angeles Dodgers",
            "LA Angels": "Los Angeles Angels",
            # Add more as needed
        }
        match = manual_mapping.get(api_team, match)
    return match

def convert_odds_to_implied_prob(odds):
    if not isinstance(odds, (int, float)) or odds == 0:
        logger.warning(f"Invalid odds value: {odds}. Returning default probability.")
        return 0.5
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)

def calculate_ev(model_prob, implied_prob, odds):
    if not isinstance(odds, (int, float)) or odds == 0:
        logger.warning(f"Invalid odds for EV calculation: {odds}. Returning 0 EV.")
        return 0.0
    decimal_odds = (odds / 100) + 1 if odds > 0 else (100 / -odds) + 1
    return (model_prob * decimal_odds) - (1 - model_prob)

def suggest_wager(model_prob, odds, base_unit=1.0, max_units=3.0):
    if not isinstance(odds, (int, float)) or odds == 0:
        logger.warning(f"Invalid odds for wager calculation: {odds}. Returning 0 units.")
        return 0.0
    decimal_odds = (odds / 100) + 1 if odds > 0 else (100 / -odds) + 1
    b = decimal_odds - 1
    q = 1 - model_prob
    kelly_fraction = ((b * model_prob) - q) / b
    kelly_fraction = max(0, kelly_fraction)
    return round(min(kelly_fraction * base_unit, max_units), 2)

def sigmoid_from_line(predicted, line, scaling=1.5):
    return np.clip(1 / (1 + np.exp(-scaling * (predicted - line))), 0.01, 0.99)

def format_output(commence_time, game, bet_type, bookmaker, odds, implied, model_prob, ev, units,
                  money_line_bet=None, total_bet=None, hits_bet=None, strikeouts_bet=None,
                  pitcher_props_bet=None, predicted_total=None):
    return {
        'Date': commence_time.strftime('%Y-%m-%d %I:%M %p'),
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
        'Predicted Total': f"{predicted_total:.2f}" if predicted_total else '',
        'Timestamp': commence_time.isoformat()
    }

def save_to_json(data, filename='output.json'):
    os.makedirs('mlb_project', exist_ok=True)
    filepath = f'mlb_project/{filename}'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved {len(data)} records to {filepath}")

def load_results():
    try:
        with open('mlb_project/output.json', 'r') as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} results from output.json")
            return data
    except Exception as e:
        logger.error(f"Error loading results: {e}")
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

def get_odds(max_retries=3, delay=5):
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    for attempt in range(max_retries):
        try:
            res = requests.get(url, params={
                'apiKey': API_KEY, 'regions': 'us',
                'bookmakers': 'fanduel,draftkings', 'markets': 'h2h,totals',
                'oddsFormat': 'american'
            }, timeout=10)
            res.raise_for_status()
            logger.info("Successfully fetched odds data.")
            remaining = res.headers.get('x-requests-remaining', 'Unknown')
            logger.info(f"API requests remaining: {remaining}")
            return res.json()
        except Exception as e:
            logger.error(f"API Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    logger.error("Failed to fetch odds data after retries.")
    return []

@cache.memoize(timeout=300)
def run_daily_predictions():
    logger.info("Starting real-time predictions.")
    odds_data = get_odds()
    results, seen = [], set()

    output_path = 'mlb_project/output.json'
    if os.path.exists(output_path):
        os.remove(output_path)
        logger.info("Cleared old output.json")

    if not odds_data:
        logger.warning("No odds data received from API. Check API key or connectivity.")
        save_to_json(results)
        return results

    logger.info(f"Processing {len(odds_data)} games from API.")
    game_df = pd.read_csv('mlb_data/mlb_games.csv')
    logger.info(f"Loaded mlb_games.csv with {len(game_df)} records.")

    unmatched_games = []
    for game in odds_data:
        try:
            home = smart_team_match(game['home_team'])
            away = smart_team_match(game['away_team'])
            logger.info(f"Processing game: {home} vs {away} (API ID: {game['id']})")

            # Handle commence_time with fallback
            try:
                utc_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
                commence_time = utc_time.astimezone(eastern_tz)
            except (KeyError, ValueError) as e:
                logger.warning(f"No valid commence_time for {home} vs {away}: {e}. Using current time.")
                commence_time = datetime.now(eastern_tz)

            row = game_df[(game_df['home_team_name'] == home) & (game_df['away_team_name'] == away)]
            if row.empty:
                logger.warning(f"No game data found for {home} vs {away} in mlb_games.csv")
                unmatched_games.append(f"{home} vs {away}")
                continue

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
                logger.debug(f"Processing bookmaker: {bm}")
                seen_points = set()
                added_ml = False

                for market in bookmaker['markets']:
                    logger.debug(f"Processing market: {market['key']} for {bm}")
                    for outcome in market['outcomes']:
                        odds = int(outcome.get('price', 0))
                        if odds == 0:
                            logger.warning(f"Skipping outcome with zero odds: {outcome}")
                            continue
                        implied = convert_odds_to_implied_prob(odds)
                        key = (game['id'], market['key'], bm, outcome.get('name'), outcome.get('point', ''))

                        if key in seen: continue
                        seen.add(key)

                        if market['key'] == 'h2h' and not added_ml:
                            model_prob = moneyline if outcome['name'] == home else 1 - moneyline
                            ev = calculate_ev(model_prob, implied, odds)
                            units = suggest_wager(model_prob, odds)
                            results.append(format_output(commence_time, f"{home} vs {away}", "Moneyline", bm, odds, implied, model_prob, ev, units, money_line_bet=home if moneyline > 0.5 else away))
                            added_ml = True

                        elif market['key'] == 'totals':
                            point = outcome.get('point')
                            if point in seen_points: continue
                            seen_points.add(point)
                            ev = calculate_ev(total_sigmoid, implied, odds)
                            units = suggest_wager(total_sigmoid, odds)
                            results.append(format_output(commence_time, f"{home} vs {away}", f"Total {point}", bm, odds, implied, total_sigmoid, ev, units, total_bet="OVER" if total_sigmoid > 0.5 else "UNDER", predicted_total=total_raw))

            results.append(format_output(commence_time, f"{home} vs {away}", "Hits Bet", "Model", 100, 0.5, hits_prob, calculate_ev(hits_prob, 0.5, 100), suggest_wager(hits_prob, 100), hits_bet="OVER" if hits_prob > 0.5 else "UNDER"))
            results.append(format_output(commence_time, f"{home} vs {away}", "Strikeouts Bet", "Model", 100, 0.5, strikeouts_prob, calculate_ev(strikeouts_prob, 0.5, 100), suggest_wager(strikeouts_prob, 100), strikeouts_bet="OVER" if strikeouts_prob > 0.5 else "UNDER"))
            results.append(format_output(commence_time, f"{home} vs {away}", "Pitcher Props Bet", "Model", 100, 0.5, pitcher_props_prob, calculate_ev(pitcher_props_prob, 0.5, 100), suggest_wager(pitcher_props_prob, 100), pitcher_props_bet=pitcher_name))
        except Exception as e:
            logger.error(f"Game error for {home} vs {away} (API ID: {game.get('id', 'unknown')}): {e}", exc_info=True)
            continue

    if unmatched_games:
        logger.error(f"Unmatched games: {', '.join(unmatched_games)}")
    save_to_json(results)
    logger.info(f"Generated {len(results)} predictions.")
    cache.set('predictions', results, timeout=300)
    return results

# --- Routes ---
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/results')
def results():
    cached_results = cache.get('predictions')
    last_updated = None
    if cached_results:
        try:
            last_updated = max(datetime.fromisoformat(r['Timestamp']) for r in cached_results)
            if datetime.now(eastern_tz) - last_updated < timedelta(minutes=5):
                logger.info("Serving cached results.")
                data = cached_results
            else:
                logger.info("Cached results stale. Fetching new data.")
                data = run_daily_predictions()
        except Exception as e:
            logger.error(f"Error processing cached results: {e}")
            data = run_daily_predictions()
    else:
        logger.info("No cached results. Fetching new data.")
        data = run_daily_predictions()

    # Filter by date to avoid timezone issues
    today = datetime.now(eastern_tz).date()
    data = [r for r in data if datetime.fromisoformat(r['Timestamp']).date() == today]
    bookmaker = request.args.get('bookmaker')
    bet_type = request.args.get('bet_type')
    if bookmaker:
        data = [r for r in data if r['Bookmaker'].lower() == bookmaker.lower()]
    if bet_type:
        data = [r for r in data if r['Bet Type'].lower().startswith(bet_type.lower())]
    return render_template("results.html", results=data, bookmakers=['fanduel', 'draftkings', 'model'],
                           bet_types=['moneyline', 'total', 'hits', 'strikeouts', 'pitcher props'])

@app.route('/top_bets')
def top_bets():
    cached_results = cache.get('predictions')
    last_updated = None
    if cached_results:
        try:
            last_updated = max(datetime.fromisoformat(r['Timestamp']) for r in cached_results)
            if datetime.now(eastern_tz) - last_updated < timedelta(minutes=5):
                logger.info("Serving cached top bets.")
                data = cached_results
            else:
                logger.info("Cached top bets stale. Fetching new data.")
                data = run_daily_predictions()
        except Exception as e:
            logger.error(f"Error processing cached top bets: {e}")
            data = run_daily_predictions()
    else:
        logger.info("No cached top bets. Fetching new data.")
        data = run_daily_predictions()

    today = datetime.now(eastern_tz).date()
    data = [r for r in data if datetime.fromisoformat(r['Timestamp']).date() == today]
    data = sorted(data, key=lambda x: float(x['EV %'].rstrip('%')), reverse=True)[:3]
    return render_template("topbets.html", results=data)

@app.route('/run-now', methods=['GET', 'POST'])
def run_now():
    if request.method == 'POST':
        run_daily_predictions()
        return redirect(url_for('results'))
    return redirect(url_for('index'))

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(str(e))
    return jsonify(error=str(e)), 500

# Start scheduler
scheduler = BackgroundScheduler(timezone=eastern_tz)
scheduler_started = False

def start_scheduler():
    global scheduler_started
    if not scheduler_started:
        try:
            scheduler.add_job(run_daily_predictions, IntervalTrigger(minutes=15))
            scheduler.start()
            scheduler_started = True
            logger.info("Scheduler started successfully for 15-minute intervals in Eastern Time.")
        except Exception as e:
            logger.error(f"Scheduler failed to start: {e}")

if __name__ == '__main__':
    start_scheduler()
    run_daily_predictions()
    app.run(debug=False)