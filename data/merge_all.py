import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FILE_AQI     = 'delhi_aqi_ready_for_merge.csv'
FILE_WEATHER = 'delhi_weather_CLEANED.csv'
FILE_FIRE    = 'daily_fire_intensity.csv'
OUTPUT_FILE  = 'Final_Model_Data.csv'

# ==============================================================================
# 1. LOAD DATASETS
# ==============================================================================
print("⏳ Loading your cleaned datasets...")
try:
    df_aqi = pd.read_csv(FILE_AQI)
    df_weather = pd.read_csv(FILE_WEATHER)
    df_fire = pd.read_csv(FILE_FIRE)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()

# ==============================================================================
# 2. STANDARDIZE TIMESTAMPS (THE FIX)
# ==============================================================================
print("🔧 Fixing Timezones...")

# --- AQI ---
# Convert to datetime
df_aqi['datetime'] = pd.to_datetime(df_aqi['datetime'])
# REMOVE Timezone info (make it naive)
df_aqi['datetime'] = df_aqi['datetime'].dt.tz_localize(None)
df_aqi.set_index('datetime', inplace=True)

# --- WEATHER ---
df_weather['date'] = pd.to_datetime(df_weather['date'])
# REMOVE Timezone info
df_weather['date'] = df_weather['date'].dt.tz_localize(None)
df_weather.set_index('date', inplace=True)

# --- FIRE ---
# Handle date column name safely
if 'DATE' in df_fire.columns:
    col = 'DATE'
elif 'acq_date' in df_fire.columns:
    col = 'acq_date'
else:
    print("❌ Error: Could not find Date column in Fire Data.")
    exit()

df_fire[col] = pd.to_datetime(df_fire[col])
# REMOVE Timezone info
df_fire[col] = df_fire[col].dt.tz_localize(None)
df_fire.set_index(col, inplace=True)

print("✅ Timezones stripped. All indexes are now compatible.")

# ==============================================================================
# 3. MERGE EVERYTHING
# ==============================================================================
print("🔗 Merging AQI and Weather (Hourly)...")
# This should now work without the TypeError
df_main = df_aqi.join(df_weather, how='inner')

print("🔥 Merging Fire Data (Daily Broadcast)...")
# Create the helper column for daily matching
df_main['date_only'] = df_main.index.normalize()
df_main = df_main.merge(df_fire[['Total_FRP']], left_on='date_only', right_index=True, how='left')

# Fill missing fire days
df_main['Total_FRP'] = df_main['Total_FRP'].fillna(0)
df_main.drop(columns=['date_only'], inplace=True)

# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================
print("🧠 Creating LSTM Features...")
df_main['AQI_Lag1'] = df_main['pm25'].shift(1)
df_main['AQI_Lag24'] = df_main['pm25'].shift(24)
df_main.dropna(inplace=True)

# ==============================================================================
# 5. SAVE
# ==============================================================================
df_main.to_csv(OUTPUT_FILE)
print("="*60)
print(f"🎉 SUCCESS! Final dataset saved as '{OUTPUT_FILE}'")
print(f"📊 Shape: {df_main.shape}")
print("="*60)