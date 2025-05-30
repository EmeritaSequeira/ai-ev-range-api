import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# Page config
st.set_page_config(page_title="EV Range Prediction App", page_icon="🚗", layout="centered")

# Load model once
@st.cache_resource
def load_model():
    return joblib.load('ev_range_predictor_reduced.pkl')

model = load_model()

# Sidebar for inputs
st.sidebar.title("Input Parameters 🚗")
SoC = st.sidebar.number_input("🔋 Current State of Charge (%)", min_value=0.0, max_value=100.0, value=80.0)
Speed = st.sidebar.number_input("🚙 Speed (Km/h)", min_value=0.0, max_value=200.0, value=60.0)
Temperature = st.sidebar.number_input("🌡️ Temperature (°C)", min_value=-20.0, max_value=60.0, value=25.0)
Terrain = st.sidebar.selectbox("🗻 Terrain Type", options=["Flat", "Hilly"])
Braking = st.sidebar.number_input("🛑 Braking (m/s²)", min_value=0.0, max_value=10.0, value=0.5)
Acceleration = st.sidebar.number_input("🏁 Acceleration (m/s²)", min_value=0.0, max_value=10.0, value=1.0)
Weather = st.sidebar.selectbox("🌦️ Weather Condition", options=["Normal", "Hot", "Cold", "Rainy"])
Prev_SoC = st.sidebar.number_input("⏪ Previous SoC (%)", min_value=0.0, max_value=100.0, value=85.0)

# Main app title
st.title("EV Range Prediction App 🚗⚡")
st.markdown("##### Predict your electric vehicle's remaining range based on driving parameters")

# Prediction button
if st.button("🔮 Predict Range"):
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
    rate = dynamic_energy_consumption_rate(Speed, Terrain, Weather)
    remaining_energy_kwh = (predicted_SoC / 100) * battery_capacity_kwh
    predicted_range_km = remaining_energy_kwh / rate

    st.success(f"✅ Predicted SoC: **{predicted_SoC:.2f}%**")
    st.info(f"📏 Estimated Remaining Range: **{predicted_range_km:.2f} km**")
    st.metric(label="Remaining Range (km)", value=f"{predicted_range_km:.1f}")

    if predicted_range_km < 50:
        st.error("⚠️ Low range detected! Please consider finding a charging station soon.")

    st.subheader("📝 Factors affecting range:")
    factors = []
    factors.append(f"✅ Speed: {'⚠️ High speed increases consumption' if Speed > 80 else '✅ Normal'}")
    factors.append(f"✅ Terrain: {'⚠️ Hilly terrain increases consumption' if Terrain == 'Hilly' else '✅ Flat'}")
    factors.append(f"✅ Temperature: {'⚠️ Hot temperature increases consumption' if Temperature > 40 else '✅ Normal'}")
    factors.append(f"✅ Weather: {'⚠️ Adverse weather increases consumption' if Weather in ['Hot','Rainy','Cold'] else '✅ Normal'}")
    for factor in factors:
        st.write(factor)
else:
    st.info("Enter input parameters in the sidebar and click 'Predict Range'")

# Show map below, independent of prediction
st.subheader("📍 Vehicle Location Map (Simulated)")
m = folium.Map(location=[12.9141, 74.8560], zoom_start=13)
folium.Marker(
    location=[12.9141, 74.8560],
    popup="EV Location",
    icon=folium.Icon(color="green", icon="car", prefix='fa')
).add_to(m)
st_folium(m, width=700, height=400)

# Footer
st.markdown("""<hr style="border: 0.5px solid #ccc;">
<div style="text-align:center; font-size: 13px;">© 2025 EV Range Prediction System | Powered by AI</div>
""", unsafe_allow_html=True)
