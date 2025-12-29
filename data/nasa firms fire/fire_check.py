import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your processed file
# Make sure this matches the filename you saved in the previous step
filename = 'daily_fire_intensity.csv' 

try:
    df = pd.read_csv(filename)
    # Ensure date column is parsed
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
    elif 'acq_date' in df.columns:
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        df.set_index('acq_date', inplace=True)
    else:
        print("❌ Error: Could not find a Date column (DATE or acq_date)")
        exit()

except FileNotFoundError:
    print(f"❌ Error: Could not find '{filename}'. Check the file name.")
    exit()

print(f"✅ Loaded {len(df)} rows of fire data.")

# 2. Check which metric you have (FRP or Count)
if 'Total_FRP' in df.columns:
    col_name = 'Total_FRP'
    print("✅ GREAT! Found 'Total_FRP'. This is the high-quality intensity data.")
elif 'fire_count' in df.columns:
    col_name = 'fire_count'
    print("⚠️ Found 'fire_count'. This is acceptable, but FRP is better.")
else:
    print(f"❌ Error: Could not find 'Total_FRP' or 'fire_count'. Columns are: {df.columns.tolist()}")
    exit()

# 3. THE VISUAL TEST (Stubble Burning Seasonality)
plt.figure(figsize=(15, 6))
df[col_name].plot(color='#ff5733', kind='area', alpha=0.6)

# Highlight Critical Periods
plt.title(f'Fire Intensity Check ({col_name})', fontsize=14)
plt.ylabel('Fire Intensity')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)

print("\n--- HOW TO PASS THIS TEST ---")
print("1. Look at the graph. Do you see TALL SPIKES in Oct/Nov of every year?")
print("2. These spikes represent the Crop Burning Season (Parali).")
print("3. If the graph is flat or random, something is wrong.")

plt.show()