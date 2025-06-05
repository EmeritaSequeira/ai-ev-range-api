import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the Statiq Mangaluru page
url = 'https://www.statiq.in/Mangaluru-ev-charging-station'

# Send a GET request to the URL
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all station entries (update the class name based on actual HTML)
stations = soup.find_all('div', class_='station-card')  # Replace with the correct class

# Initialize a list to store the data
data = []

for station in stations:
    name = station.find('h3').text.strip()  # Replace with the correct tag
    address = station.find('p', class_='address').text.strip()  # Replace with the correct class
    charger_type = station.find('span', class_='charger-type').text.strip()  # Replace with the correct class
    availability = station.find('span', class_='availability').text.strip()  # Replace with the correct class

    data.append({
        'Station Name': name,
        'Address': address,
        'Charger Type': charger_type,
        'Availability': availability
    })

# Create a DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('mangalore_ev_stations.csv', index=False)
