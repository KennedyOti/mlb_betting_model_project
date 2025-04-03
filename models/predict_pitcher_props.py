import joblib
import numpy as np
import os

# Load model and encoder
model_path = "models/pitcher_props_model.pkl"
encoder_path = "models/opponent_encoder.pkl"

model = joblib.load(model_path)
le_opponent = joblib.load(encoder_path)

# Input prompt
print("⚾ Enter match details for Pitcher Props prediction:\n")
# Ask for pitcher name
pitcher_name = input("👤 Pitcher Name: ")

opponent_team = input("🆚 Opponent Team: ")
innings_pitched = float(input("⏱️ Innings Pitched: "))
rolling_so_avg = float(input("📊 Rolling SO Average: "))
career_games = int(input("📈 Career Games: "))

# Encode opponent
if opponent_team not in le_opponent.classes_:
    print(f"❌ Error: '{opponent_team}' not recognized in encoder.")
    print("📌 Please make sure you use one of the trained team names.")
    exit()

opponent_encoded = le_opponent.transform([opponent_team])[0]

# Build input for prediction
input_data = np.array([[innings_pitched, rolling_so_avg, career_games, opponent_encoded]])

# Predict
predicted_so = model.predict(input_data)[0]

# Output
print(f"\n🎯 Predicted Strikeouts by {pitcher_name} vs {opponent_team}: {predicted_so:.2f}")
