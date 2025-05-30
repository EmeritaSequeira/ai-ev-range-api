import streamlit as st
import pandas as pd
import requests
import joblib
import folium
from streamlit_folium import st_folium

# Load your trained model
model = joblib.load("ev_range_predictor_reduced.pkl")

# Your ORS API key
ORS_API_KEY = '5b3ce3597851110001cf6248d8fa62413ac90656b012ab35066ba6f5c5c19369f7811fe6695fb97d'

def geocode_place(place):
    url = f'https://api.openrouteservice.org/geocode/search'
    headers = {'Authorization': ORS_API_KEY}
    params = {'text': place, 'size': 1}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json()['features']
        if results:
            coords = results[0]['geometry']['coordinates']
            return coords[1], coords[0]  # lat, lon
    return None, None

def get_route(start, end):
    url = 'https://api.openrouteservice.org/v2/directions/driving-car'
    headers = {'Authorization': ORS_API_KEY}
    params = {'start': f'{start[1]},{start[0]}', 'end': f'{end[1]},{end[0]}'}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        geometry = data['features'][0]['geometry']['coordinates']
        distance = data['features'][0]['properties']['segments'][0]['distance'] / 1000
        return geometry, distance
    return None, None

def dynamic_energy_rate(speed, terrain, weather):
    rate = 0.15
    if speed <= 50:
        rate = 0.12
    elif speed > 80:
        rate = 0.18
    if terrain.lower() == 'hilly':
        rate *= 1.2
    if weather.lower() == 'hot':
        rate *= 1.1
    return rate

# Streamlit App
st.title("🚗 EV Route Assistant with Range Prediction")
st.markdown("Enter place names (e.g., Kankanady to State Bank)")

# --- INPUT SECTION ---
start_place = st.text_input("Start Place", value="Kankanady, Mangalore")
end_place = st.text_input("End Place", value="State Bank, Mangalore")

speed = st.slider('Average Speed (Km/h)', 10, 120, 60)
terrain = st.selectbox('Terrain Type', ['Flat', 'Hilly'])
weather = st.selectbox('Weather Condition', ['Clear', 'Hot', 'Rainy', 'Cold'])
prev_soc = st.slider('Previous SoC (%)', 0, 100, 75)
braking = st.number_input('Braking (m/s²)', 0.0, 5.0, 0.1)
acceleration = st.number_input('Acceleration (m/s²)', 0.0, 5.0, 0.1)

if st.button("🔮 Predict"):
    start_lat, start_lon = geocode_place(start_place)
    end_lat, end_lon = geocode_place(end_place)

    if not start_lat or not end_lat:
        st.error("Could not locate the places. Please enter valid names.")
    else:
        start = (start_lat, start_lon)
        end = (end_lat, end_lon)
        geometry, distance = get_route(start, end)

        if geometry:
            input_df = pd.DataFrame([{
                'Speed (Km/h)': speed,
                'Terrain': terrain,
                'Weather': weather,
                'Prev_SoC': prev_soc,
                'Braking (m/s²)': braking,
                'Acceleration (m/s²)': acceleration
            }])

            predicted_soc = model.predict(input_df)[0]
            battery_kwh = 40
            rate = dynamic_energy_rate(speed, terrain, weather)
            remaining_kwh = (predicted_soc / 100) * battery_kwh
            predicted_range = remaining_kwh / rate

            st.session_state['results'] = {
                'distance': distance,
                'soc': predicted_soc,
                'range': predicted_range,
                'geometry': geometry,
                'start': start,
                'end': end
            }
        else:
            st.warning("Unable to retrieve route information.")

# --- RESULTS ---
if 'results' in st.session_state:
    results = st.session_state['results']
    st.subheader("🚦 Trip Summary")
    st.write(f"Route Distance: **{results['distance']:.2f} km**")
    st.write(f"Predicted SoC after trip: **{results['soc']:.2f}%**")
    st.write(f"Predicted Driving Range: **{results['range']:.2f} km**")

    m = folium.Map(location=results['start'], zoom_start=13)
    folium.Marker(results['start'], tooltip='Start', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(results['end'], tooltip='End', icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine([(lat, lon) for lon, lat in results['geometry']], color='blue').add_to(m)
    st_folium(m, width=700, height=500)
