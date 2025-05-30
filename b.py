import joblib
import pandas as pd

# Load your saved pipeline
model = joblib.load('ev_range_predictor_reduced.pkl')

# Inputs
speed = 60
terrain_type = 'Flat'
weather = 'Clear'
prev_soc = 75
braking = 0.1
acceleration = 0.1

input_data = pd.DataFrame([{
    'Speed (Km/h)': speed,
    'Terrain': terrain_type,
    'Weather': weather,
    'Prev_SoC': prev_soc,
    'Braking (m/s²)': braking,
    'Acceleration (m/s²)': acceleration
}])

predicted_soc = model.predict(input_data)[0]

def dynamic_energy_consumption_rate(speed_kmh, terrain, weather):
    rate = 0.15
    if speed_kmh <= 50:
        rate = 0.12
    elif speed_kmh > 80:
        rate = 0.18
    if terrain == 'Hilly':
        rate *= 1.2
    if weather == 'Hot':
        rate *= 1.1
    return rate

battery_capacity_kwh = 40
rate = dynamic_energy_consumption_rate(speed, terrain_type, weather)
remaining_energy_kwh = (predicted_soc / 100) * battery_capacity_kwh
predicted_range_km = remaining_energy_kwh / rate

print(f"Predicted SoC: {predicted_soc:.2f} %")
print(f"Predicted Range: {predicted_range_km:.2f} km")
