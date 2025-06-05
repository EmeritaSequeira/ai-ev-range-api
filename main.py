import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(page_title="EV Range Prediction Dashboard", layout="wide")

# Title & Description
st.title("🔋 AI-Powered EV Range Prediction Dashboard")
st.markdown("""
Use this dashboard to visualize **predicted vs actual EV range**, battery status, and driving patterns.
""")

# Upload Data
uploaded_file = st.file_uploader("📂 Upload your EV prediction dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Preprocess datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Sidebar filters
    vehicle_list = df['Vehicle_ID'].unique()
    selected_vehicle = st.sidebar.selectbox("Select Vehicle ID", vehicle_list)

    vehicle_df = df[df['Vehicle_ID'] == selected_vehicle]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("🔋 Initial SoC (%)", f"{vehicle_df['SoC (%)'].iloc[0]:.1f}")
    col2.metric("📏 Distance Covered (km)", f"{vehicle_df['Distance (Km)'].sum():.1f}")
    col3.metric("⚡️ Final SoC (%)", f"{vehicle_df['SoC (%)'].iloc[-1]:.1f}")

    # SoC Over Time
    fig_soc = px.line(vehicle_df, x='Timestamp', y='SoC (%)', title="Battery SoC (%) Over Time", markers=True)
    st.plotly_chart(fig_soc, use_container_width=True)

    # Predicted vs Actual Range (if present)
    if 'Predicted_Range' in df.columns and 'Actual_Range' in df.columns:
        fig_range = px.line(vehicle_df, x='Timestamp', y=['Predicted_Range', 'Actual_Range'],
                            title="Predicted vs Actual Range", markers=True)
        st.plotly_chart(fig_range, use_container_width=True)

    # Speed over time (optional)
    if 'Speed (Km/h)' in df.columns:
        fig_speed = px.line(vehicle_df, x='Timestamp', y='Speed (Km/h)', title="Speed Over Time", markers=True)
        st.plotly_chart(fig_speed, use_container_width=True)

    # Low SoC Alert
    low_soc_rows = vehicle_df[vehicle_df['SoC (%)'] < 20]
    if not low_soc_rows.empty:
        st.warning(f"⚠️ {len(low_soc_rows)} low battery alerts detected (SoC < 20%) for vehicle {selected_vehicle}.")

else:
    st.info("📊 Please upload a CSV file to get started.")

# Footer (Updated)
st.markdown("""
<hr style="border: 1px solid #ccc;">
<div style="text-align:center; font-size: 13px;">
Built by Emerita Sequeira | <a href="https://github.com/emeritasequeira/ai-ev-range-api" target="_blank">GitHub Repo</a>
</div>
""", unsafe_allow_html=True)
