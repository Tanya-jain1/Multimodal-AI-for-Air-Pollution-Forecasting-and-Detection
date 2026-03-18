import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('delhi_weather_CLEANED.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("--- WEATHER DATA HEALTH REPORT ---")

# CHECK 1: Missing Values
missing = df.isnull().sum().sum()
print(f"1. Total Missing Values: {missing}")
if missing > 0:
    print("   ⚠️ WARNING: You still have gaps! Run interpolation again.")
else:
    print("   ✅ CLEAN: No empty cells found.")

# CHECK 2: Physical Range Checks (Delhi Specific)
# Temperature: Rarely below 2°C or above 50°C
temp_issues = df[(df['Temperature'] < 2) | (df['Temperature'] > 50)]
print(f"2. Temperature Outliers (<2°C or >50°C): {len(temp_issues)} hours")

# Humidity: Must be 0-100%
hum_issues = df[(df['Humidity'] < 0) | (df['Humidity'] > 100)]
print(f"3. Humidity Errors (0-100%): {len(hum_issues)} hours")

# Wind Speed: Rarely above 100 km/h in Delhi (unless storm)
wind_issues = df[df['wind_speed_10m'] > 100] if 'wind_speed_10m' in df.columns else []
print(f"4. Extreme Wind Events (>100km/h): {len(wind_issues)} hours")

print("-" * 30)
# VISUALIZATION

# ---------------------------------------------------------
# Plot 1: Temperature Seasonality (The "Wave")
# ---------------------------------------------------------
plt.figure(figsize=(15, 4))
df['Temperature'].plot(color='#d62728', title='1. Temperature Check: Do you see yearly Summer/Winter waves?')
plt.ylabel('Temp (°C)')
plt.tight_layout()
plt.show()  # This will display the first plot

# ---------------------------------------------------------
# Plot 2: Monsoon Check (Rain & Humidity)
# ---------------------------------------------------------
# We use subplots here just to get the 'ax1' object for the twin axes
fig, ax1 = plt.subplots(figsize=(15, 4))

# Humidity on the primary y-axis
df['Humidity'].rolling(24).mean().plot(ax=ax1, color='blue', alpha=0.6, label='Humidity (24h Avg)')
ax1.set_ylabel('Humidity (%)')
ax1.set_title('2. Monsoon Check: Does Humidity/Rain spike every July-Sept?')
ax1.legend(loc='upper left')

# Rain on the secondary y-axis
ax2 = ax1.twinx()
df['Rain'].resample('W').sum().plot(kind='area', ax=ax2, color='green', alpha=0.3, label='Weekly Rain')
ax2.set_ylabel('Rain (mm)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()  # This will display the second plot

# ---------------------------------------------------------
# Plot 3: Wind Vectors (The "Blob")
# ---------------------------------------------------------
# A square figure (8x8) is best here so the circular wind blob isn't stretched into an oval
plt.figure(figsize=(8, 8))
plt.scatter(df['Wind_X'], df['Wind_Y'], alpha=0.1, s=1, color='purple')
plt.title('3. Wind Vector Integrity: Should look like a circular blob')
plt.xlabel('Wind X (East-West)')
plt.ylabel('Wind Y (North-South)')
plt.grid(True)

# Draw crosshairs
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

plt.tight_layout()
plt.show()  # This will display the third plot