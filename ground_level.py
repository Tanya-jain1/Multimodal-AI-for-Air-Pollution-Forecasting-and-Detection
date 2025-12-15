import kagglehub
import os
import pandas as pd
import numpy as np

# ===================================================================
# Configuration
# ===================================================================
# Define constants for easier modification
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]
LAG_DAYS = 7
ROLLING_WINDOW = 7
OUTPUT_FILE = "delhi_aq_featured_daily.csv"

# ===================================================================
# 1. Load and Clean Data
# ===================================================================
print("🚀 Starting data preparation...")

# Download the dataset from Kaggle and get the path to its folder
path = kagglehub.dataset_download("deepaksirohiwal/delhi-air-quality")
csv_file = os.path.join(path, "delhi_aqi.csv")
print(f"Dataset downloaded to: {path}")

# Read the CSV file
df = pd.read_csv(csv_file)

# Convert 'date' to datetime and set it as the index
df['datetime'] = pd.to_datetime(df['date'])
df = df.set_index('datetime').drop(columns=['date'])

# Rename columns to a standard format for consistency
rename_dict = {"pm2_5": "pm25"}
df = df.rename(columns=rename_dict)

# Select only the pollutant columns we need
# The 'inplace=True' argument ensures we keep only columns that exist in the dataframe
df = df[df.columns.intersection(POLLUTANTS)]

# Fill any missing values using the last known value (forward fill)
df.fillna(method="ffill", inplace=True)

print("✅ Data loading and cleaning complete.")
print("Initial data shape:", df.shape)
print("Preview of cleaned data:\n", df.head())


# ===================================================================
# 2. Resample and Engineer Features
# ===================================================================
print("\n🚀 Starting feature engineering...")

# Resample the hourly data to daily averages
daily_df = df.resample("D").mean()

# Interpolate to fill any gaps that might have been created during resampling
daily_df = daily_df.interpolate(method='linear')

# --- Efficient Feature Creation Loop ---
for pollutant in POLLUTANTS:
    # Check if the pollutant column exists in our daily dataframe
    if pollutant in daily_df.columns:
        # Create lag features (values from previous days)
        for lag in range(1, LAG_DAYS + 1):
            daily_df[f"{pollutant}_lag{lag}"] = daily_df[pollutant].shift(lag)

        # Create rolling window features (average and std dev over a period)
        rolling_stats = daily_df[pollutant].rolling(window=ROLLING_WINDOW)
        daily_df[f"{pollutant}_{ROLLING_WINDOW}d_avg"] = rolling_stats.mean()
        daily_df[f"{pollutant}_{ROLLING_WINDOW}d_std"] = rolling_stats.std()

# Drop rows with NaN values, which are created by the lag/rolling features at the start of the dataset
daily_df.dropna(inplace=True)

print("✅ Feature engineering complete.")
print("Final shape after adding features:", daily_df.shape)
print("Preview of data with new features:\n", daily_df.head())


# ===================================================================
# 3. Save Final Dataset
# ===================================================================
daily_df.to_csv(OUTPUT_FILE)
print(f"\n✅ Successfully saved the final dataset as '{OUTPUT_FILE}'")