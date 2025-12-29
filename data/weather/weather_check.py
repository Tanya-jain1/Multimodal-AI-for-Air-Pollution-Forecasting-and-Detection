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
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Temperature Seasonality (The "Wave")
df['Temperature'].plot(ax=axes[0], color='#d62728', title='1. Temperature Check: Do you see yearly Summer/Winter waves?')
axes[0].set_ylabel('Temp (°C)')

# Plot 2: Monsoon Check (Rain & Humidity)
# We plot Humidity in Blue and Rain as bars in Green
ax2 = axes[1]
df['Humidity'].rolling(24).mean().plot(ax=ax2, color='blue', alpha=0.6, label='Humidity (24h Avg)')
ax2_twin = ax2.twinx()
df['Rain'].resample('W').sum().plot(kind='area', ax=ax2_twin, color='green', alpha=0.3, label='Weekly Rain')
ax2.set_title('2. Monsoon Check: Does Humidity/Rain spike every July-Sept?')
ax2.legend(loc='upper left')
ax2_twin.set_ylabel('Rain (mm)')

# Plot 3: Wind Vectors (The "Blob")
# If Wind_X and Wind_Y are correct, a scatter plot should look like a circular "blob" or "star"
# If it looks like a straight line or a box, something is wrong.
axes[2].scatter(df['Wind_X'], df['Wind_Y'], alpha=0.1, s=1, color='purple')
axes[2].set_title('3. Wind Vector Integrity: Should look like a circular blob (not a line!)')
axes[2].set_xlabel('Wind X (East-West)')
axes[2].set_ylabel('Wind Y (North-South)')
axes[2].grid(True)
# Draw crosshairs
axes[2].axhline(0, color='black', lw=1)
axes[2].axvline(0, color='black', lw=1)

plt.tight_layout()
plt.show()