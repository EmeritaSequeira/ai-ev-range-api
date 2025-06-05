import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# === Page Setup ===
st.set_page_config(page_title="EV Range Predictor Dashboard", page_icon="🚗", layout="wide")

# === Load Model ===
@st.cache_resource
def load_model():
    return joblib.load("ev_range_predictor_reduced.pkl")

model = load_model()

# === Sidebar - Inputs ===
st.sidebar.header("📥 Input Parameters")

speed = st.sidebar.slider("🚙 Speed (Km/h)", 0.0, 200.0, 60.0)
acceleration = st.sidebar.slider("🏁 Acceleration (m/s²)", 0.0, 10.0, 1.5)
braking = st.sidebar.slider("🛑 Braking (m/s²)", 0.0, 10.0, 0.8)
prev_soc = st.sidebar.slider("⏪ Previous SoC (%)", 0.0, 100.0, 85.0)
temperature = st.sidebar.slider("🌡️ Temperature (°C)", -20.0, 60.0, 25.0)
soc = st.sidebar.slider("🔋 Current SoC (%)", 0.0, 100.0, 80.0)
terrain = st.sidebar.selectbox("🗻 Terrain", ["Flat", "Hilly"])
weather = st.sidebar.selectbox("🌦️ Weather", ["Normal", "Hot", "Cold", "Rainy"])

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh every 30 seconds", value=False)
show_download = st.sidebar.checkbox("⬇️ Show CSV Download", value=True)

if auto_refresh:
    st_autorefresh(interval=30 * 1000, key="autorefresh")

# === Main Title ===
st.title("🚗 Smart EV Range Prediction Dashboard")
st.markdown(
    """
    <style>
    .big-font {
        font-size:22px !important;
        font-weight:600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='big-font'>Estimate your <b>State of Charge (SoC)</b> and <b>Range</b> based on driving & environmental conditions.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# === Predict Button ===
if st.button("🔮 Predict Range"):

    # Prepare input for model
    input_data = pd.DataFrame([{
        "Speed (Km/h)": speed,
        "Acceleration (m/s²)": acceleration,
        "Braking (m/s²)": braking,
        "Prev_SoC": prev_soc,
        "Terrain": terrain,
        "Weather": weather
    }])

    predicted_soc = model.predict(input_data)[0]

    # Energy consumption rate calculation (heuristic)
    def energy_rate(speed, terrain, weather):
        rate = 0.15
        if speed <= 50:
            rate = 0.12
        elif speed > 80:
            rate = 0.18
        if terrain == "Hilly":
            rate *= 1.2
        if weather in ["Hot", "Cold", "Rainy"]:
            rate *= 1.1
        return rate

    battery_capacity = 40  # kWh nominal
    consumption_rate = energy_rate(speed, terrain, weather)
    remaining_energy = (predicted_soc / 100) * battery_capacity
    predicted_range = remaining_energy / consumption_rate
    soc_drop = prev_soc - predicted_soc

    # === Metrics Section ===
    st.header("📊 Prediction Summary")
    metric_cols = st.columns(3)
    metric_cols[0].metric(label="🔋 Predicted SoC", value=f"{predicted_soc:.2f}%")
    metric_cols[1].metric(label="📏 Estimated Range", value=f"{predicted_range:.2f} Km")
    metric_cols[2].metric(label="⬇️ SoC Drop", value=f"{soc_drop:.2f}%")

    st.markdown("---")

    # === Visualization Section ===
    st.subheader("Visualizations")
    vis_cols = st.columns(3)

    # SoC Drop Gauge
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
    vis_cols[0].plotly_chart(fig_soc, use_container_width=True)

    # Battery Usage Pie Chart
    labels = ['Used Capacity (kWh)', 'Remaining Capacity (kWh)']
    values = [battery_capacity - remaining_energy, remaining_energy]
    fig_pie = px.pie(names=labels, values=values, color_discrete_sequence=["#ff9999", "#00bfff"])
    vis_cols[1].plotly_chart(fig_pie, use_container_width=True)

    # Speed vs Energy Consumption Line Chart
    speeds = list(range(20, 130, 10))
    consumption_rates = [energy_rate(s, terrain, weather) for s in speeds]
    df_curve = pd.DataFrame({'Speed (Km/h)': speeds, 'Consumption Rate (kWh/km)': consumption_rates})
    fig_line = px.line(df_curve, x='Speed (Km/h)', y='Consumption Rate (kWh/km)', markers=True,
                       color_discrete_sequence=["#0073e6"])
    vis_cols[2].plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # === Efficiency Score ===
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
    st.header("⚙️ Driving Efficiency Score")
    st.progress(eff_score / 100)
    st.info(f"Your driving efficiency is **{eff_score}%**")

    # === Input Summary and Tips in Expanders ===
    with st.expander("📝 Input Summary"):
        st.markdown(f"""
        - **Speed**: {speed} Km/h  
        - **Acceleration**: {acceleration} m/s²  
        - **Braking**: {braking} m/s²  
        - **Terrain**: {terrain}  
        - **Weather**: {weather}  
        - **Temperature**: {temperature}°C  
        """)

    with st.expander("💡 Personalized Driving Tips"):
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

    # === CSV Download ===
    if show_download:
        st.markdown("---")
        st.header("📂 Download Prediction")
        download_df = pd.DataFrame([{
            "Predicted SoC (%)": predicted_soc,
            "Predicted Range (Km)": predicted_range,
            "SoC Drop (%)": soc_drop,
            "Driving Efficiency (%)": eff_score,
        }])
        st.download_button(
            label="Download CSV",
            data=download_df.to_csv(index=False),
            file_name="ev_prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("🔹 Please enter input parameters from the sidebar and click **Predict Range** to get started.")

# === Footer ===
st.markdown(
    """
    <hr style="border: 1px solid #ccc; margin-top: 3rem; margin-bottom: 1rem;">
    <div style="text-align:center; font-size: 13px; color: #666;">
    🚗 Built with ❤️ by Emerita Sequeira | 
    <a href="https://github.com/emeritasequeira/ai-ev-range-api" target="_blank" style="text-decoration:none; color:#0066cc;">
    GitHub Repo
    </a>
    </div>
    """,
    unsafe_allow_html=True
)
