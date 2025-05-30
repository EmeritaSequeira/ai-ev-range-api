import joblib
import pandas as pd

# Load the saved reduced model pipeline
model = joblib.load('ev_range_predictor_reduced.pkl')

# Example input data (adding 'Weather' and 'Prev_SoC' columns)
soc = 80  # Example SoC in percentage
speed = 60  # Example speed in km/h
temperature = 25  # Example temperature in Celsius
terrain_type = 'Flat'  # Example terrain type ('Flat', 'Hilly', etc.)
braking = 0.1  # Example braking value (m/s²)
acceleration = 0.1  # Example acceleration value (m/s²)

# Add missing columns with dummy values
weather = 'Clear'  # Example weather condition
prev_soc = 75  # Example previous SoC value (you can adjust this)

# Prepare the input as a DataFrame, ensuring all required columns are included
input_data = pd.DataFrame([{
    'SoC': soc,
    'Speed (Km/h)': speed,
    'Temperature': temperature,
    'Terrain': terrain_type,
    'Braking (m/s²)': braking,
    'Acceleration (m/s²)': acceleration,
    'Weather': weather,  # Added Weather column
    'Prev_SoC': prev_soc  # Added Prev_SoC column
}])

# Make prediction
predicted_SoC = model.predict(input_data)[0]

# Calculate predicted range based on SoC and dynamic energy consumption rate
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
# Example weather condition for calculation
weather_condition = 'Clear'  # Add your weather condition here

# Calculate predicted range
rate = dynamic_energy_consumption_rate(speed, terrain_type, weather_condition)
remaining_energy_kwh = (predicted_SoC / 100) * battery_capacity_kwh
predicted_range_km = remaining_energy_kwh / rate

# Print the results
print(f"Predicted SoC: {predicted_SoC:.2f} %")
print(f"Predicted Range: {predicted_range_km:.2f} km")
