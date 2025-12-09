# ===============================
# Delhi Weather Data Download & Processing
# ===============================
import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# -------------------------------
# Step 1: Download dataset using kagglehub
# -------------------------------
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("yug201/daily-climate-time-series-data-delhi-india")
print("Dataset downloaded to:", dataset_path)

# -------------------------------
# Step 2: Locate CSV file
# -------------------------------
csv_file = None
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        csv_file = os.path.join(dataset_path, file)
        break

if csv_file is None:
    raise FileNotFoundError("No CSV file found in the downloaded dataset!")

print("Found CSV file:", csv_file)

# -------------------------------
# Step 3: Load CSV into pandas
# -------------------------------
df = pd.read_csv(csv_file)
print("First 5 rows of raw dataset:\n", df.head())

# -------------------------------
# Step 4: Convert DATE column to datetime
# -------------------------------
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# -------------------------------
# Step 5: Handle missing values safely
# -------------------------------
# Forward-fill all columns
df = df.ffill()

# Interpolate only numeric columns to avoid warnings
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].interpolate()

# -------------------------------
# Step 6: Clip unrealistic values
# -------------------------------
if 'temp' in df.columns:
    df['temp'] = df['temp'].clip(-5, 50)

if 'humidity' in df.columns:
    df['humidity'] = df['humidity'].clip(0, 100)

if 'windspeed' in df.columns:
    df['windspeed'] = df['windspeed'].clip(0, 50)

if 'precip' in df.columns:
    df['precip'] = df['precip'].clip(0, 200)

if 'sealevelpressure' in df.columns:
    df['sealevelpressure'] = df['sealevelpressure'].clip(900, 1100)

# -------------------------------
# Step 7: Feature engineering - date parts
# -------------------------------
df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month
df['Day'] = df['DATE'].dt.day
df['WeekOfYear'] = df['DATE'].dt.isocalendar().week
df['DayOfWeek'] = df['DATE'].dt.dayofweek  # Monday=0
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# -------------------------------
# Step 8: Rolling statistics (7-day average)
# -------------------------------
df['temp_7d_avg'] = df['temp'].rolling(window=7, min_periods=1).mean()
df['humidity_7d_avg'] = df['humidity'].rolling(window=7, min_periods=1).mean()
df['windspeed_7d_avg'] = df['windspeed'].rolling(window=7, min_periods=1).mean()
df['precip_7d_avg'] = df['precip'].rolling(window=7, min_periods=1).mean()

# -------------------------------
# Step 9: Lag features (previous day)
# -------------------------------
df['temp_lag1'] = df['temp'].shift(1)
df['humidity_lag1'] = df['humidity'].shift(1)
df['windspeed_lag1'] = df['windspeed'].shift(1)
df['precip_lag1'] = df['precip'].shift(1)

# Fill any NaNs created by rolling/lag
df = df.bfill()

# -------------------------------
# Step 10: Scale numeric features
# -------------------------------
features = ['temp', 'humidity', 'windspeed', 'precip', 'sealevelpressure',
            'temp_7d_avg', 'humidity_7d_avg', 'windspeed_7d_avg', 'precip_7d_avg',
            'temp_lag1', 'humidity_lag1', 'windspeed_lag1', 'precip_lag1']

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# -------------------------------
# Step 11: Save processed dataset
# -------------------------------
output_file = "delhi_weather_processed.csv"
df.to_csv(output_file, index=False)
print(f"✅ Processed dataset saved as '{output_file}'")
print(df.head())