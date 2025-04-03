import pandas as pd
import joblib
import numpy as np

# Load the model and encoder
model_path = "models/total_runs_model.pkl"
encoder_path = "models/team_encoder.pkl"

model = joblib.load(model_path)
le_team = joblib.load(encoder_path)

# Team names (full names as seen in the data)
home_team = "Boston Red Sox"
away_team = "New York Yankees"

# Encode teams
home_encoded = le_team.transform([home_team])[0]
away_encoded = le_team.transform([away_team])[0]

# Prepare input DataFrame
input_data = pd.DataFrame([{
    'home_team_encoded': home_encoded,
    'away_team_encoded': away_encoded
}])

# Make prediction
predicted_total = model.predict(input_data)[0]

# Ask for line input
try:
    line_input = input("Enter O/U line (e.g., 8.5) or leave blank for default 7.5: ")
    over_under_line = float(line_input) if line_input else 8.5
except ValueError:
    over_under_line = 7.5

# Determine suggested bet
if predicted_total > over_under_line:
    suggestion = f"OVER (vs line of {over_under_line})"
else:
    suggestion = f"UNDER (vs line of {over_under_line})"

# Output result
print(f"\n📊 Predicted Total Runs for {home_team} vs {away_team}: {predicted_total:.2f}")
print(f"💡 Suggested Bet: {suggestion}")
