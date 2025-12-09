import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ===================================================================
# Configuration
# ===================================================================
AQ_DATA_FILE = "delhi_aq_featured_daily.csv"
WEATHER_DATA_FILE = "delhi_weather_processed.csv"
TARGET_POLLUTANT = "pm25"
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1

# ===================================================================
# 1. Load and Merge Data
# ===================================================================
print("🚀 Step 1: Loading and merging data...")

# Load datasets
aq_df = pd.read_csv(AQ_DATA_FILE)
weather_df = pd.read_csv(WEATHER_DATA_FILE)

# Convert date columns to datetime objects and set as index
aq_df['datetime'] = pd.to_datetime(aq_df['datetime'])
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
aq_df = aq_df.set_index('datetime')
weather_df = weather_df.set_index('DATE')

# Merge the two dataframes
df_merged = aq_df.join(weather_df, how='inner')
print(f"Merged data shape: {df_merged.shape}")
print("Merged data preview:\n", df_merged.head())

# ===================================================================
# 2. Final Data Preparation for Modeling
# ===================================================================
print("\n🚀 Step 2: Preparing data for time series modeling...")

# Select only numeric columns for modeling
df_numeric = df_merged.select_dtypes(include=np.number)

# Reorder columns to have the target variable first
cols = [TARGET_POLLUTANT] + [col for col in df_numeric.columns if col != TARGET_POLLUTANT]
df_final = df_numeric[cols]

# Scale all features to a range between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_final)

# Create a separate scaler for the target variable to inverse transform predictions later
target_scaler = MinMaxScaler()
target_scaler.fit(df_final[[TARGET_POLLUTANT]])

# Function to create sequences of data for time series forecasting
def create_sequences(data, seq_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length + forecast_horizon - 1, 0]) # Target is the first column
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQUENCE_LENGTH, FORECAST_HORIZON)
print(f"Shape of X (sequences, timesteps, features): {X.shape}")
print(f"Shape of y (labels): {y.shape}")

# Split data into training, validation, and test sets (80-10-10 split)
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# ===================================================================
# 3. Build and Train LSTM Model
# ===================================================================
print("\n🚀 Step 3: Building and training the LSTM model...")

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.summary()

# Callback to stop training early if validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history_lstm = lstm_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

print("✅ LSTM model training complete.")

# ===================================================================
# 4. Evaluate and Visualize LSTM Model
# ===================================================================
print("\n🚀 Step 4: Evaluating the LSTM model...")

# Make predictions on the test set
predictions_lstm_scaled = lstm_model.predict(X_test)

# Inverse scale the predictions and actual values to their original range
predictions_lstm = target_scaler.inverse_transform(predictions_lstm_scaled)
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate and print performance metrics
mae = mean_absolute_error(y_test_actual, predictions_lstm)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_lstm))
print(f"--- LSTM Model Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")

# Plot the results
print("📊 Generating plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 7))
plt.title(f'LSTM Model: Actual vs. Predicted {TARGET_POLLUTANT}', fontsize=16)

# Plotting both actual and predicted values as lines
plt.plot(y_test_actual, label='Actual Values', color='blue', linewidth=2)
plt.plot(predictions_lstm, label='LSTM Predictions', color='orangered', linewidth=2, linestyle='--')

plt.xlabel('Time (Days in Test Set)')
plt.ylabel(f'{TARGET_POLLUTANT} Concentration')
plt.legend()
plt.show()