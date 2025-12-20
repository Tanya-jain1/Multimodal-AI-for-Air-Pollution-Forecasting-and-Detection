import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Input, Flatten, Dropout
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

# CNN-LSTM benefits from slightly longer windows to find patterns
TIME_STEPS = 30  
EPOCHS = 50
BATCH_SIZE = 32

# ===================================================================
# 1. Load and Merge Data
# ===================================================================
print(" 🚀  Step 1: Loading and merging data...")

try:
    aq_df = pd.read_csv(AQ_DATA_FILE, parse_dates=True, index_col='datetime')
    weather_df = pd.read_csv(WEATHER_DATA_FILE, parse_dates=True, index_col='DATE')
    df_merged = aq_df.join(weather_df, how='inner')
    print(" ✅  Data merged. Shape:", df_merged.shape)
except FileNotFoundError:
    print(" ❌  Error: Files not found. Run the cleaning scripts first!")
    exit()

# ===================================================================
# 2. Feature Selection
# ===================================================================
# Select raw features only
selected_columns = [
    'pm25', 'pm10', 'no2', 'so2', 'co', 'o3',
    'temp', 'humidity', 'windspeed', 'precip', 'sealevelpressure'
]
existing_cols = [c for c in selected_columns if c in df_merged.columns]
df_model = df_merged[existing_cols].copy()

# Safety clipping
if 'pm25' in df_model.columns: df_model['pm25'] = df_model['pm25'].clip(upper=600)
if 'co' in df_model.columns: df_model['co'] = df_model['co'].clip(upper=2000)
df_model = df_model.fillna(method='ffill').fillna(method='bfill')

# ===================================================================
# 3. Split and Scale (Correct Order)
# ===================================================================
print("\n 🚀  Step 3: Splitting and Scaling...")

train_size = int(len(df_model) * 0.8)
train_df = df_model.iloc[:train_size]
test_df = df_model.iloc[train_size:]

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit on TRAIN
X_train_scaled = feature_scaler.fit_transform(train_df)
y_train_scaled = target_scaler.fit_transform(train_df[[TARGET_VARIABLE]])

# Transform TEST
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

print(f" ✅  Sequences ready. Input shape: {X_train.shape}")

# ===================================================================
# 5. Build Hybrid CNN-LSTM Model
# ===================================================================
print("\n 🚀  Step 5: Building CNN-LSTM Model...")

input_shape = (X_train.shape[1], X_train.shape[2]) 
input_layer = Input(shape=input_shape)

# --- CNN Block (Extracts short-term features) ---
# Filters: 64 (detects 64 different types of patterns)
# Kernel Size: 3 (looks at 3-day windows)
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)

# Pooling: Reduces noise and dimensionality
# IMPORTANT: Pool size is small (2) to not lose too much info
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

# --- LSTM Block (Learns time dependencies from CNN features) ---
# We use fewer units (50) because CNN has already simplified the data
x = LSTM(50, activation='relu', return_sequences=False)(x)
x = Dropout(0.2)(x)

# --- Output Block ---
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

y_pred_scaled = model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_actual = target_scaler.inverse_transform(y_test)

mse = np.mean((y_actual - y_pred)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_actual - y_pred))

print("\n==================================")
print(f" CNN-LSTM RESULTS FOR {TARGET_VARIABLE}")
print("==================================")
print(f" RMSE: {rmse:.2f}")
print(f" MAE:  {mae:.2f}")
print("==================================")

plt.figure(figsize=(15, 6))
plt.plot(y_actual, label='Actual', color='blue', alpha=0.6)
plt.plot(y_pred, label='Predicted (CNN-LSTM)', color='purple', linestyle='--')
plt.title(f'Corrected CNN-LSTM Model: {TARGET_VARIABLE} Prediction')
plt.legend()
plt.savefig("cnn_lstm_corrected_results.png")
print(" ✅  Plot saved as 'cnn_lstm_corrected_results.png'")
plt.show()