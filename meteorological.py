import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ===================================================================
# Configuration
# ===================================================================
OUTPUT_FILE = "delhi_weather_with_visibility.csv"
START_DATE = "2013-01-01"
END_DATE = "2024-12-30"  # Adjust to today's date if needed
LATITUDE = 28.6139
LONGITUDE = 77.2090

# ===================================================================
# 1. Fetch Data from Open-Meteo
# ===================================================================
print(" 🚀  Fetching historical weather data (including Visibility)...")

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Define request parameters
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": LATITUDE,
	"longitude": LONGITUDE,
	"start_date": START_DATE,
	"end_date": END_DATE,
	# We specifically request 'visibility' along with other key metrics
	"daily": ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean", "visibility_mean"],
	"timezone": "Asia/Kolkata"
}

# Make the API call
responses = openmeteo.weather_api(url, params=params)
response = responses[0]

# ===================================================================
# 2. Process the Response into a DataFrame
# ===================================================================
daily = response.Daily()
daily_data = {
    "date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
}

# Extract values using the variable indices
daily_data["temp"] = daily.Variables(0).ValuesAsNumpy()
daily_data["precip"] = daily.Variables(1).ValuesAsNumpy()
daily_data["windspeed"] = daily.Variables(2).ValuesAsNumpy()
daily_data["humidity"] = daily.Variables(3).ValuesAsNumpy()
daily_data["visibility"] = daily.Variables(4).ValuesAsNumpy() # Raw Visibility in meters

df = pd.DataFrame(data=daily_data)

# Rename 'date' to 'DATE' to match your existing pipeline
df = df.rename(columns={"date": "DATE"})
df['DATE'] = pd.to_datetime(df['DATE']).dt.date # Remove timezone info for easier merging

# ===================================================================
# 3. Feature Engineering: The "Inverse Visibility"
# ===================================================================
print(" 🛠️  Engineering PBL Proxy Feature...")

# Handle missing visibility values (forward fill)
df['visibility'] = df['visibility'].fillna(method='ffill').fillna(method='bfill')

# Create Inverse Visibility
# Formula: 1 / (Visibility_in_km + 0.1)
# We convert meters to km first for scale stability
df['visibility_km'] = df['visibility'] / 1000.0
df['inverse_visibility'] = 1.0 / (df['visibility_km'] + 0.1)

# Drop intermediate column if you want, or keep it
# df = df.drop(columns=['visibility_km'])

# ===================================================================
# 4. Standard Cleaning & Scaling (Matching your old script)
# ===================================================================
print(" 🧹  Cleaning and Scaling...")

# Rolling Averages (7-day)
cols_to_roll = ['temp', 'humidity', 'windspeed', 'precip', 'inverse_visibility']
for col in cols_to_roll:
    df[f'{col}_7d_avg'] = df[col].rolling(window=7, min_periods=1).mean()
    df[f'{col}_lag1'] = df[col].shift(1)

df = df.fillna(method='bfill')

# Scale (Optional here, but good if you want ready-to-use data)
# Note: Usually better to scale inside the model script to avoid leakage, 
# but matching your request structure:
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save
df.to_csv(OUTPUT_FILE, index=False)
print(f" ✅  Saved new weather data with PBL proxy to: {OUTPUT_FILE}")
print(df[['DATE', 'inverse_visibility']].head())