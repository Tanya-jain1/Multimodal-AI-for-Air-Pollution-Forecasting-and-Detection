import pandas as pd

# 1. Load your weather file
filename = 'delhi_weather_engineered.csv'  # Update if your file name is different
df = pd.read_csv(filename)

# 2. Drop the EMPTY 925hPa columns
cols_to_drop = [
    'wind_speed_925hPa', 
    'wind_direction_925hPa', 
    'wind_925_u', 
    'wind_925_v'
]
print(f"Dropping empty columns: {cols_to_drop}")
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# 3. Rename columns to standard names
# This ensures your weather data matches the naming convention for the AI model
rename_map = {
    'temperature_2m': 'Temperature',
    'relative_humidity_2m': 'Humidity',
    'precipitation': 'Rain',
    'boundary_layer_height': 'PBL_Height',
    'ventilation_coeff': 'Ventilation',
    # Note: Your file already has 'Wind_X' and 'Wind_Y', so we keep them as is.
}
df.rename(columns=rename_map, inplace=True)

# 4. Set Time Index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 5. Select only the features we need
# We leave out 'shortwave_radiation' etc unless you specifically want them.
final_cols = ['Temperature', 'Humidity', 'Rain', 'PBL_Height', 'Wind_X', 'Wind_Y', 'Ventilation']
df_clean = df[final_cols]

# 6. Fill any small remaining gaps (Linear Interpolation)
df_clean = df_clean.interpolate(method='time')

# 7. Save
df_clean.to_csv('delhi_weather_CLEANED.csv')
print("\nSuccess! Saved clean weather data to 'delhi_weather_CLEANED.csv'")
print(df_clean.head())