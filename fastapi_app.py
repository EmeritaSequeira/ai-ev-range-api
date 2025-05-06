from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('ev_range_predictor_reduced.pkl')

app = FastAPI()

# Define the input data structure
class PredictionRequest(BaseModel):
    SoC: float
    Speed: float
    Temperature: float
    Terrain: str
    Braking: float
    Acceleration: float
    Weather: str
    Prev_SoC: float

# Prediction endpoint
@app.post('/predict')
def predict(data: PredictionRequest):
    input_data = pd.DataFrame([{
        'SoC': data.SoC,
        'Speed (Km/h)': data.Speed,
        'Temperature': data.Temperature,
        'Terrain': data.Terrain,
        'Braking (m/s²)': data.Braking,
        'Acceleration (m/s²)': data.Acceleration,
        'Weather': data.Weather,
        'Prev_SoC': data.Prev_SoC
    }])

    predicted_SoC = model.predict(input_data)[0]

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
    rate = dynamic_energy_consumption_rate(data.Speed, data.Terrain, data.Weather)
    remaining_energy_kwh = (predicted_SoC / 100) * battery_capacity_kwh
    predicted_range_km = remaining_energy_kwh / rate

    return {"Predicted_SoC": predicted_SoC, "Predicted_Range_km": predicted_range_km}
