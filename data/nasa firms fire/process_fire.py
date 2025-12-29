import pandas as pd
import numpy as np

# ===================================================================
# Configuration
# ===================================================================
ARCHIVE_FILE = "fire_archive2016.csv"  
NRT_FILE     = "fire_nrt2016.csv"      
OUTPUT_FILE  = "daily_fire_intensity.csv" # Renamed to reflect 'Intensity'

START_DATE = "2016-01-01" 
END_DATE   = "2025-12-23"

# ===================================================================
# 1. Load Both Files
# ===================================================================
print("🔥 Loading NASA FIRMS data...")
try:
    df_archive = pd.read_csv(ARCHIVE_FILE)
    df_nrt = pd.read_csv(NRT_FILE)
    print(f"   - Archive: {len(df_archive)} rows")
    print(f"   - NRT:     {len(df_nrt)} rows")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()

# ===================================================================
# 2. Merge and Clean
# ===================================================================
print("🔗 Merging Archive and NRT data...")
df = pd.concat([df_archive, df_nrt], ignore_index=True)

# Standardize column names to lowercase (NASA sometimes mixes case)
df.columns = [c.lower() for c in df.columns]

df['acq_date'] = pd.to_datetime(df['acq_date'])
df = df.sort_values(by='acq_date')

# Remove Duplicates
initial_count = len(df)
df = df.drop_duplicates(subset=['latitude', 'longitude', 'acq_date', 'acq_time'])
print(f"   - Dropped {initial_count - len(df)} duplicates.")

# ===================================================================
# 3. Filter Low Confidence
# ===================================================================
if 'confidence' in df.columns:
    # Convert confidence to string to handle 'l', 'n', 'h' mixed with numbers
    df = df[df['confidence'].astype(str).isin(['n', 'h', 'nominal', 'high'])]
    print(f"   - Filtered low confidence. Remaining: {len(df)}")

# ===================================================================
# 4. Aggregate (THE UPGRADE)
# ===================================================================
print("📊 Aggregating Fire Intensity (FRP)...")

# OLD WAY: Count fires
# daily_fires = df.groupby('acq_date').size().reset_index(name='fire_count')

# NEW WAY: Sum the Heat Energy (FRP)
# This creates a 'Total_FRP' column
daily_fires = df.groupby('acq_date')['frp'].sum().reset_index(name='Total_FRP')
daily_fires = daily_fires.set_index('acq_date')

# ===================================================================
# 5. Reindex (Handle "Zero Fire" Days)
# ===================================================================
full_date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
daily_fires = daily_fires.reindex(full_date_range, fill_value=0)
daily_fires.index.name = 'DATE'

# ===================================================================
# 6. Save
# ===================================================================
daily_fires.to_csv(OUTPUT_FILE)
print(f"✅ Saved '{OUTPUT_FILE}' with Total_FRP column.")
print(daily_fires.tail())