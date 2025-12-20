import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Input, Dropout, Attention, GlobalAveragePooling1D, Concatenate, LayerNormalization
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

TIME_STEPS = 30
EPOCHS = 50
BATCH_SIZE = 32

# ===================================================================
# 1. Load and Merge Data (Standard Pipeline)
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
# 3. Split and Scale
# ===================================================================
print("\n 🚀  Step 3: Splitting and Scaling...")
train_size = int(len(df_model) * 0.8)
train_df = df_model.iloc[:train_size]
test_df = df_model.iloc[train_size:]

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(train_df)
y_train_scaled = target_scaler.fit_transform(train_df[[TARGET_VARIABLE]])

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

# ===================================================================
# 5. Build CNN-LSTM-Attention Model
# ===================================================================
print("\n 🚀  Step 5: Building CNN-LSTM with Attention...")

input_shape = (X_train.shape[1], X_train.shape[2]) 
inputs = Input(shape=input_shape)

# --- 1. CNN Block (Feature Extraction) ---
# Extracts local short-term patterns (e.g., daily spikes)
x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

# --- 2. LSTM Block (Sequence Learning) ---
# return_sequences=True is CRITICAL here. 
# It keeps the time dimension so Attention can look at each step.
lstm_out = LSTM(64, return_sequences=True, activation='relu')(x)
lstm_out = LayerNormalization()(lstm_out) # Stabilizes training

# --- 3. Attention Mechanism ---
# Self-attention: Query and Value are both the LSTM output.
# The model asks "Which parts of my own history are important?"
# We use a built-in Keras Attention layer.
attention_out = Attention()([lstm_out, lstm_out])

# --- 4. Fusion & Output ---
# We combine the LSTM memory with the Attention focus
x = Concatenate()([lstm_out, attention_out])

# Flatten the timeline to a single vector
x = GlobalAveragePooling1D()(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
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

print("\n===========================================")
print(f" CNN-LSTM-ATTENTION RESULTS FOR {TARGET_VARIABLE}")
print("===========================================")
print(f" RMSE: {rmse:.2f}")
print(f" MAE:  {mae:.2f}")
print("===========================================")

plt.figure(figsize=(15, 6))
plt.plot(y_actual, label='Actual', color='blue', alpha=0.6)
plt.plot(y_pred, label='Predicted (Attention)', color='darkorange', linestyle='--')
plt.title(f'CNN-LSTM with Attention: {TARGET_VARIABLE} Prediction')
plt.legend()
plt.savefig("cnn_lstm_attention_results.png")
print(" ✅  Plot saved as 'cnn_lstm_attention_results.png'")
plt.show()