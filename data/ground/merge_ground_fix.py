import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the file you just saved
df = pd.read_csv('delhi_aqi_FINAL_fixed.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# 2. THE FIX: Clip Extreme Highs
# We replace anything > 1200 with NaN, then interpolate (smooth it out)
# This removes the 25,000 spike and the 2,000 blocks.
upper_limit = 1200
print(f"Values above {upper_limit}: {(df['pm25'] > upper_limit).sum()}")

# Mask outliers as NaN
df['pm25'] = df['pm25'].mask(df['pm25'] > upper_limit)

# Fill the holes we just made using interpolation
df['pm25'] = df['pm25'].interpolate(method='time')

# 3. Save the TRULY Cleaned file
df.to_csv('delhi_aqi_ready_for_merge.csv')
print("Saved! Your data is now statistically valid.")

# 4. Final Verification Plot
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['pm25'])
plt.title("Final Cleaned Data (Capped at 1200)")
plt.ylabel("PM2.5")
plt.show()