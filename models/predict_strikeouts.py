import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load model and encoders
model = joblib.load("models/strikeouts_model.pkl")
le_team = joblib.load("models/team_encoder.pkl")
le_opponent = joblib.load("models/opponent_encoder.pkl")
le_venue = joblib.load("models/venue_encoder.pkl")

print("🎯 Enter match details for Strikeout prediction:\n")

# Collect user input
team = input("🏟️ Team: ").strip()
opponent = input("🆚 Opponent Team: ").strip()
venue = input("📍Venue: ").strip()
innings_pitched = float(input("⚾ Innings Pitched: "))
bb = int(input("🔴 Walks (BB): "))
er = int(input("🔥 Earned Runs (ER): "))
h = int(input("⚾ Hits Allowed (H): "))
r = int(input("📉 Runs Allowed (R): "))

# Encode categorical inputs
team_encoded = le_team.transform([team])[0]
opponent_encoded = le_opponent.transform([opponent])[0]
venue_encoded = le_venue.transform([venue])[0]

# Prepare input for model
input_data = np.array([[team_encoded, opponent_encoded, venue_encoded, innings_pitched, bb, er, h, r]])

# Predict
predicted_so = model.predict(input_data)[0]

# Output
print(f"\n🎯 Predicted Strikeouts for {team} vs {opponent} at {venue}: {predicted_so:.2f}")
