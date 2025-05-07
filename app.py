import streamlit as st
import pandas as pd
import joblib

# Load the saved model (cache to prevent reloading every rerun)
@st.cache_resource
def load_model():
    return joblib.load('ev_range_predictor_reduced.pkl')

model = load_model()

# Streamlit app
st.title("EV Range Prediction App 🚗⚡")

st.subheader("Enter input parameters:")

# Input fields
SoC = st.number_input("State of Charge (SoC) (%)", min_value=0.0, max_value=100.0, value=80.0)
Speed = st.number_input("Speed (Km/h)", min_value=0.0, max_value=200.0, value=60.0)
Temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=25.0)
Terrain = st.selectbox("Terrain Type", options=["Flat", "Hilly"])
Braking = st.number_input("Braking (m/s²)", min_value=0.0, max_value=10.0, value=0.5)
Acceleration = st.number_input("Acceleration (m/s²)", min_value=0.0, max_value=10.0, value=1.0)
Weather = st.selectbox("Weather Condition", options=["Normal", "Hot", "Cold", "Rainy"])
Prev_SoC = st.number_input("Previous SoC (%)", min_value=0.0, max_value=100.0, value=85.0)

if st.button("Predict Range"):
    # Prepare input DataFrame
    input_data = pd.DataFrame([{
        'SoC': SoC,
        'Speed (Km/h)': Speed,
        'Temperature': Temperature,
        'Terrain': Terrain,
        'Braking (m/s²)': Braking,
        'Acceleration (m/s²)': Acceleration,
        'Weather': Weather,
        'Prev_SoC': Prev_SoC
    }])

    # Prediction
    predicted_SoC = model.predict(input_data)[0]

    # Energy consumption function
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

    battery_capacity_kwh = 40  # Example capacity
    rate = dynamic_energy_consumption_rate(Speed, Terrain, Weather)
    remaining_energy_kwh = (predicted_SoC / 100) * battery_capacity_kwh
    predicted_range_km = remaining_energy_kwh / rate

    st.success(f"✅ Predicted SoC: **{predicted_SoC:.2f}%**")
    st.info(f"📏 Estimated Remaining Range: **{predicted_range_km:.2f} km**")

    # Optional visualization
    st.progress(predicted_SoC / 100)
    st.metric(label="Remaining Range (km)", value=f"{predicted_range_km:.2f}")

