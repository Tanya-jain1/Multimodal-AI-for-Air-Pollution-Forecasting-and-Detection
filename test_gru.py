import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# ===================================================================
# Configuration
# ===================================================================
AQ_DATA_FILE = "delhi_aq_cleaned_raw.csv"
WEATHER_DATA_FILE = "delhi_weather_cleaned_raw.csv"
TARGET_VARIABLE = "pm25"

# IMPROVEMENT: Increased from 10 to 30 to capture monthly trends
TIME_STEPS = 30  
EPOCHS = 50
BATCH_SIZE = 32

# ===================================================================
# 1. Load and Merge Data
# ===================================================================
print(" 🚀  Step 1: Loading and merging data...")

try:
    # Load data
    aq_df = pd.read_csv(AQ_DATA_FILE, parse_dates=True, index_col='datetime')
    weather_df = pd.read_csv(WEATHER_DATA_FILE, parse_dates=True, index_col='DATE')
    
    # Merge on index
    df_merged = aq_df.join(weather_df, how='inner')
    print(" ✅  Data merged. Shape:", df_merged.shape)

except FileNotFoundError:
    print(" ❌  Error: Files not found. Check your paths.")
    exit()

# ===================================================================
# 2. Clean and Feature Selection (FIXING REDUNDANCY)
# ===================================================================
print("\n 🚀  Step 2: Selecting features and handling outliers...")

# FIX 1: Explicitly select ONLY raw features. 
# We DROP 'lag1', '7d_avg' etc. because GRU creates its own internal memory.
# Feeding "lags of lags" creates noise.
selected_columns = [
    'pm25', 'pm10', 'no2', 'so2', 'co', 'o3',  # Pollutants
    'temp', 'humidity', 'windspeed', 'precip'  # Weather
]

# Keep only existing columns from the list
existing_cols = [c for c in selected_columns if c in df_merged.columns]
df_model = df_merged[existing_cols].copy()

# FIX 2: Handle Outliers (Clipping)
# Extreme values (like CO > 6000) squash the MinMaxScaler, making normal variations invisible.
print("    - Clipping outliers...")
if 'pm25' in df_model.columns:
    df_model['pm25'] = df_model['pm25'].clip(upper=600)  # Clip hazardous levels
if 'pm10' in df_model.columns:
    df_model['pm10'] = df_model['pm10'].clip(upper=900)
if 'co' in df_model.columns:
    df_model['co'] = df_model['co'].clip(upper=2000) # Clip extreme CO spikes

# Handle missing values if any remain
df_model = df_model.fillna(method='ffill').fillna(method='bfill')

# ===================================================================
# 3. Split and Scale (FIXING DATA LEAKAGE)
# ===================================================================
print("\n 🚀  Step 3: Splitting and Scaling (Correctly)...")

# FIX 3: Split BEFORE Scaling
# This prevents the model from "peeking" at the test set's min/max values.
train_size = int(len(df_model) * 0.8)
train_df = df_model.iloc[:train_size]
test_df = df_model.iloc[train_size:]

print(f"    - Training samples: {len(train_df)}")
print(f"    - Testing samples: {len(test_df)}")

# Define Scalers
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit ONLY on Training Data
print("    - Fitting scalers on training data...")
X_train_scaled = feature_scaler.fit_transform(train_df)
y_train_scaled = target_scaler.fit_transform(train_df[[TARGET_VARIABLE]])

# Transform Test Data using the Training Scaler
X_test_scaled = feature_scaler.transform(test_df)
y_test_scaled = target_scaler.transform(test_df[[TARGET_VARIABLE]])

# ===================================================================
# 4. Create Sequences
# ===================================================================
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, TIME_STEPS)

print(f" ✅  Sequences created. X_train shape: {X_train.shape}")

# ===================================================================
# 5. Build GRU Model
# ===================================================================
print("\n 🚀  Step 5: Building GRU Model...")

input_shape = (X_train.shape[1], X_train.shape[2]) # (TIME_STEPS, features)
input_layer = Input(shape=input_shape)

# IMPROVEMENT: Added Dropout to prevent overfitting
x = GRU(128, return_sequences=False, activation='relu')(input_layer)
x = Dropout(0.2)(x) 
output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# ===================================================================
# 6. Train
# ===================================================================
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# ===================================================================
# 7. Evaluate
# ===================================================================
print("\n 🚀  Step 7: Evaluating...")

# Predict
y_pred_scaled = model.predict(X_test)

# Inverse Transform
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_actual = target_scaler.inverse_transform(y_test)

# Calculate Metrics
mse = np.mean((y_actual - y_pred)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_actual - y_pred))

print("\n==================================")
print(f" RESULTS FOR {TARGET_VARIABLE}")
print("==================================")
print(f" RMSE: {rmse:.2f}")
print(f" MAE:  {mae:.2f}")
print("==================================")

# Plot
plt.figure(figsize=(15, 6))
plt.plot(y_actual, label='Actual', color='blue', alpha=0.6)
plt.plot(y_pred, label='Predicted (GRU)', color='red', linestyle='--')
plt.title(f'Corrected GRU Model: {TARGET_VARIABLE} Prediction')
plt.legend()
plt.savefig("gru_corrected_results.png")
print(" ✅  Plot saved as 'gru_corrected_results.png'")
plt.show()