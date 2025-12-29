import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry

# 1. Setup Client
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# 2. Coordinates (Central Delhi)
coords = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "start_date": "2016-01-01",
    "end_date": "2025-12-23"
}

# 3. Request Variables (Added Solar Radiation)
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": coords["latitude"],
    "longitude": coords["longitude"],
    "start_date": coords["start_date"],
    "end_date": coords["end_date"],
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "boundary_layer_height",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_speed_925hPa",    # Transport Wind (Stubble)
        "wind_direction_925hPa",
        "shortwave_radiation"   # ADDED: Crucial for Ozone/Smog formation
    ],
    "timezone": "Asia/Kolkata"
}

responses = openmeteo.weather_api(url, params=params)
response = responses[0]

# 4. Process Data
hourly = response.Hourly()
hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
}

for i, col in enumerate(params["hourly"]):
    hourly_data[col] = hourly.Variables(i).ValuesAsNumpy()

df_weather = pd.DataFrame(data=hourly_data)

# --- CRITICAL FIXES START HERE ---

# Fix 1: Sync Timezone to IST (Matches CPCB Data)
df_weather['date'] = df_weather['date'].dt.tz_convert('Asia/Kolkata')
# Optional: Remove timezone info to match generic CSV formats, but keep the IST time
df_weather['date'] = df_weather['date'].dt.tz_localize(None) 

# Fix 2: Vectorize Wind (Cyclical Encoding)
# We convert Speed(Mag) and Dir(Angle) into U(x) and V(y) components
# This helps the LSTM understand that North (360) and North (1) are the same.

# Surface Winds (10m)
wd_rad = np.deg2rad(df_weather['wind_direction_10m'])
df_weather['wind_10m_u'] = df_weather['wind_speed_10m'] * np.cos(wd_rad)
df_weather['wind_10m_v'] = df_weather['wind_speed_10m'] * np.sin(wd_rad)

# Transport Winds (925hPa)
wd_925_rad = np.deg2rad(df_weather['wind_direction_925hPa'])
df_weather['wind_925_u'] = df_weather['wind_speed_925hPa'] * np.cos(wd_925_rad)
df_weather['wind_925_v'] = df_weather['wind_speed_925hPa'] * np.sin(wd_925_rad)

# Fix 3: Advanced Feature Engineering
# Ventilation Coefficient (The "Pollution Trap" Metric)
df_weather['ventilation_coeff'] = df_weather['boundary_layer_height'] * df_weather['wind_speed_10m']

# Drop raw wind columns (optional, but recommended for Training to reduce noise)
# df_weather.drop(columns=['wind_speed_10m', 'wind_direction_10m', ...], inplace=True)

# --- FIXES END ---

print(df_weather.head())
df_weather.to_csv("delhi_weather_engineered.csv", index=False)