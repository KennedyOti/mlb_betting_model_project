import joblib
import numpy as np
import os

# Load model and encoders
model_path = "models/hits_model.pkl"
venue_encoder_path = "models/venue_encoder.pkl"
opponent_encoder_path = "models/opponent_encoder.pkl"

model = joblib.load(model_path)
le_venue = joblib.load(venue_encoder_path)
le_opponent = joblib.load(opponent_encoder_path)

# 📝 Prompt user for input
print("🔍 Enter match details for Hits prediction:\n")

team = input("📍Team: ")
opponent = input("🆚 Opponent Team: ")
venue = input("📍Venue: ")
innings_pitched = float(input("⚾ Innings Pitched: "))
walks = float(input("🔴 Walks (BB): "))
earned_runs = float(input("🔥 Earned Runs (ER): "))
hits = float(input("💥 Hits Allowed (H): "))
runs = float(input("🏃 Runs Allowed (R): "))

# 🎯 Encode venue and opponent
opponent_encoded = le_opponent.transform([opponent])[0]
venue_encoded = le_venue.transform([venue])[0]

# 📦 Build input array
input_data = np.array([[opponent_encoded, venue_encoded, innings_pitched, walks, earned_runs, hits, runs]])

# 🔮 Make prediction
predicted_hits = model.predict(input_data)[0]

# 📊 Display result
print(f"\n🔢 Predicted Hits Allowed for {team} vs {opponent} at {venue}: {predicted_hits:.2f}")

ou_line = input("Enter Over/Under line (e.g., 5.5): ") or "5.5"
ou_line = float(ou_line)

print()  # spacing

if predicted_hits > ou_line:
    print(f"📈 Prediction: OVER {ou_line} (Predicted: {predicted_hits:.2f})")
else:
    print(f"📉 Prediction: UNDER {ou_line} (Predicted: {predicted_hits:.2f})")
