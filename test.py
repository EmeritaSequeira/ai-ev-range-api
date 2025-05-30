import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ✅ Page setup
st.set_page_config(page_title="EV Range Predictor", page_icon="🚗", layout="centered")

# ✅ Load trained model
@st.cache_resource
def load_model():
    return joblib.load("ev_range_predictor_reduced.pkl")

model = load_model()

# ✅ App Title
st.title("🔋 Smart EV Range Prediction")
st.markdown("Estimate your **State of Charge (SoC)** and **range** based on driving and environmental conditions.")

# ✅ Input Panel
st.subheader("📥 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    speed = st.slider("🚙 Speed (Km/h)", 0.0, 200.0, 60.0)
    acceleration = st.slider("🏁 Acceleration (m/s²)", 0.0, 10.0, 1.5)
    terrain = st.selectbox("🗻 Terrain", ["Flat", "Hilly"])
    weather = st.selectbox("🌦️ Weather", ["Normal", "Hot", "Cold", "Rainy"])

with col2:
    braking = st.slider("🛑 Braking (m/s²)", 0.0, 10.0, 0.8)
    prev_soc = st.slider("⏪ Previous SoC (%)", 0.0, 100.0, 85.0)
    temperature = st.slider("🌡️ Temperature (°C)", -20.0, 60.0, 25.0)
    soc = st.slider("🔋 Current SoC (%)", 0.0, 100.0, 80.0)

# ✅ Predict Button
if st.button("🔮 Predict Range"):

    # Prepare input for the model
    input_data = pd.DataFrame([{
        "Speed (Km/h)": speed,
        "Acceleration (m/s²)": acceleration,
        "Braking (m/s²)": braking,
        "Prev_SoC": prev_soc,
        "Terrain": terrain,
        "Weather": weather
    }])

    # Prediction
    predicted_soc = model.predict(input_data)[0]

    # Energy consumption logic
    def energy_rate(speed, terrain, weather):
        rate = 0.15
        if speed <= 50: rate = 0.12
        elif speed > 80: rate = 0.18
        if terrain == "Hilly": rate *= 1.2
        if weather in ["Hot", "Cold", "Rainy"]: rate *= 1.1
        return rate

    battery_capacity = 40  # in kWh
    consumption_rate = energy_rate(speed, terrain, weather)
    remaining_energy = (predicted_soc / 100) * battery_capacity
    predicted_range = remaining_energy / consumption_rate

    # ✅ Display SoC and Range
    st.metric(label="🔋 Predicted SoC", value=f"{predicted_soc:.2f}%")
    st.metric(label="📏 Estimated Range", value=f"{predicted_range:.2f} Km")

    # ✅ SoC Drop Gauge Chart
    soc_drop = prev_soc - predicted_soc
    st.subheader("📉 SoC Drop Indicator")
    fig_soc = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soc_drop,
        title={"text": "SoC Drop (%)"},
        gauge={
            'axis': {'range': [0, 50]},
            'bar': {'color': "deepskyblue"},
            'steps': [
                {'range': [0, 15], 'color': "#cceeff"},
                {'range': [15, 30], 'color': "#80d4ff"},
                {'range': [30, 50], 'color': "#3399ff"},
            ]
        }
    ))
    st.plotly_chart(fig_soc, use_container_width=True)

    # ✅ Range Breakdown Pie Chart
    st.subheader("🔋 Battery Usage Distribution")
    labels = ['Used Capacity (kWh)', 'Remaining Capacity (kWh)']
    values = [battery_capacity - remaining_energy, remaining_energy]
    fig_pie = px.pie(names=labels, values=values, color_discrete_sequence=["#ff9999", "#00bfff"])
    st.plotly_chart(fig_pie, use_container_width=True)

    # ✅ Optional: Speed vs Consumption Curve
    st.subheader("📈 Speed vs Energy Consumption (Simulated)")
    speeds = list(range(20, 130, 10))
    consumption_rates = [energy_rate(s, terrain, weather) for s in speeds]
    df_curve = pd.DataFrame({'Speed (Km/h)': speeds, 'Consumption Rate (kWh/km)': consumption_rates})
    fig_line = px.line(df_curve, x='Speed (Km/h)', y='Consumption Rate (kWh/km)',
                       markers=True, color_discrete_sequence=["#0073e6"])
    st.plotly_chart(fig_line, use_container_width=True)

    # ✅ Efficiency Score
    def efficiency(speed, acc, brake, terrain, weather):
        score = 100
        if speed > 100: score -= 15
        elif speed > 80: score -= 10
        if speed < 30: score -= 5
        if acc > 3: score -= 10
        if brake > 3: score -= 10
        if terrain == "Hilly": score -= 5
        if weather in ["Hot", "Cold", "Rainy"]: score -= 5
        return max(score, 0)

    eff_score = efficiency(speed, acceleration, braking, terrain, weather)
    st.subheader("⚙️ Driving Efficiency Score")
    st.progress(eff_score)
    st.info(f"Your driving efficiency is **{eff_score}%**")

    # ✅ Conditions Summary
    st.subheader("📝 Summary of Inputs")
    st.markdown(f"""
    - **Speed**: {speed} Km/h  
    - **Acceleration**: {acceleration} m/s²  
    - **Braking**: {braking} m/s²  
    - **Terrain**: {terrain}  
    - **Weather**: {weather}  
    - **Temperature**: {temperature}°C  
    """)

    # ✅ Tips Section
    st.subheader("💡 Personalized Driving Tips")
    tips = []
    if speed > 100: tips.append("🔻 Reduce your speed for better efficiency.")
    if acceleration > 3: tips.append("⚠️ Avoid harsh acceleration.")
    if braking > 3: tips.append("🛑 Use smooth braking to conserve energy.")
    if terrain == "Hilly": tips.append("⛰️ Avoid hilly roads when possible.")
    if weather in ["Hot", "Cold", "Rainy"]: tips.append("❄️ Adjust AC/heater wisely.")

    if tips:
        for tip in tips:
            st.warning(tip)
    else:
        st.success("✅ Great conditions for optimal range!")

# ✅ Footer
st.markdown("""
<hr style="border: 1px solid #ccc;">
<div style="text-align:center; font-size: 13px;">
© 2025 EV Range Predictor | Built with ❤️ using Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
