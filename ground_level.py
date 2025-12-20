import kagglehub
import os
import pandas as pd
import numpy as np

# ===================================================================
# Configuration
# ===================================================================
OUTPUT_FILE = "delhi_aq_cleaned_raw.csv" # Renamed to indicate it's RAW data
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# ===================================================================
# 1. Load Data
# ===================================================================
print(" 🚀  Starting data preparation...")
path = kagglehub.dataset_download("deepaksirohiwal/delhi-air-quality")
csv_file = os.path.join(path, "delhi_aqi.csv")

df = pd.read_csv(csv_file)
df['datetime'] = pd.to_datetime(df['date'])
df = df.set_index('datetime').drop(columns=['date'])

# Rename columns
rename_dict = {"pm2_5": "pm25"}
df = df.rename(columns=rename_dict)

# Select columns
df = df[df.columns.intersection(POLLUTANTS)]

# ===================================================================
# 2. Outlier Removal (NEW & CRITICAL)
# ===================================================================
# We clip extreme values to preventing them from ruining the scaler later.
print(" ✂️   Clipping extreme outliers...")
if 'pm25' in df.columns: df['pm25'] = df['pm25'].clip(upper=600)
if 'pm10' in df.columns: df['pm10'] = df['pm10'].clip(upper=900)
if 'co' in df.columns:   df['co'] = df['co'].clip(upper=5000) 
if 'so2' in df.columns:  df['so2'] = df['so2'].clip(upper=200)
if 'no2' in df.columns:  df['no2'] = df['no2'].clip(upper=300)

# ===================================================================
# 3. Resample and Save
# ===================================================================
# Resample to daily averages
daily_df = df.resample("D").mean()

# Fill missing values
daily_df = daily_df.interpolate(method='linear').fillna(method='bfill')

# Save the CLEAN RAW data (No Scaling, No Lags)
daily_df.to_csv(OUTPUT_FILE)
print(f" ✅  Successfully saved cleaned raw data to '{OUTPUT_FILE}'")
print(daily_df.head())