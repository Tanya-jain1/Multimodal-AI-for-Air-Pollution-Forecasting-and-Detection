import pandas as pd
import glob
import os

# 1. Setup: Define path to your folder containing the CSVs
path = r'D:\vs code\Multimodal AI for Air Pollution Forecasting and Detection\csv\csv_ground'  # Change this to your actual folder path
all_files = glob.glob(os.path.join(path, "*.csv"))

print(f"Found {len(all_files)} files. Processing...")

processed_dfs = []

for filename in all_files:
    # 2. Load the individual file
    # We explicitly tell pandas to treat 'pm25' as numeric, converting errors to NaN
    df = pd.read_csv(filename)
    
    # 3. Standardization (The most important part)
    # Ensure column names are clean (lowercase, stripped of spaces)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Check if required columns exist
    if 'datetime' not in df.columns or 'pm25' not in df.columns:
        print(f"Skipping {filename}: Missing datetime or pm25 column")
        continue

    # Convert to datetime object
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    
    # Set time as the index (required for resampling)
    df.set_index('datetime', inplace=True)
    
    # 4. RESAMPLING: The fix for "sensor changed/frequented"
    # This forces the data to be 'Hourly' ('1H'). 
    # If a sensor fired 4 times in an hour, it takes the mean.
    # If it didn't fire, it creates a row with NaN (which we handle later).
    df_hourly = df['pm25'].resample('1H').mean()
    
    # Store this cleaned series
    processed_dfs.append(df_hourly)

# 5. Merge Strategy: Combine all stations
# We concat them side-by-side first to see all stations per hour
print("Merging all stations...")
combined_df = pd.concat(processed_dfs, axis=1)

# 6. Create the "City-Wide Average"
# We calculate the mean across all columns (stations) for every hour.
# This handles the issue where sensors change over years. 
# In 2016, it might average 3 stations. In 2024, it might average 20.
# The result is a single continuous timeline for Delhi.
delhi_master_aqi = combined_df.mean(axis=1)

# Convert back to a DataFrame
final_df = pd.DataFrame(delhi_master_aqi, columns=['pm25'])

# 7. Final Cleanup
# Handle small gaps (e.g., power outage for 2 hours) using Linear Interpolation
final_df['pm25'] = final_df['pm25'].interpolate(method='time', limit=24) # Limit fill to 24h gaps

# Drop any remaining NaNs (e.g., usually at the very start or end if data is missing)
final_df.dropna(inplace=True)

# 8. Save
final_df.to_csv('delhi_aqi_cleaned_2016_2025.csv')
print("Success! Saved as 'delhi_aqi_cleaned_2016_2025.csv'")
print(final_df.head())