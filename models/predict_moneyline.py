import joblib
import numpy as np

# Load model and encoder
model = joblib.load("models/moneyline_model.pkl")
le_team = joblib.load("models/team_encoder.pkl")

# Prompt user for inputs
print("⚾ Enter match teams for Moneyline prediction:\n")
home_team = input("🏠 Home Team: ").strip()
away_team = input("🆚 Away Team: ").strip()

# Encode teams
try:
    home_encoded = le_team.transform([home_team])[0]
    away_encoded = le_team.transform([away_team])[0]
except ValueError as e:
    print(f"🚫 Error: One or both teams not recognized.\nDetails: {e}")
    exit()

# Prepare input
input_data = np.array([[home_encoded, away_encoded]])

# Make prediction
pred = model.predict(input_data)[0]

# Display result
print("\n🎯 Prediction Result:")
if pred == 1:
    print(f"✅ Predicted Winner: {home_team} (Home)")
else:
    print(f"❌ Predicted Winner: {away_team} (Away)")
