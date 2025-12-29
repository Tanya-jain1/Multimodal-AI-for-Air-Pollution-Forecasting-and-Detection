import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Conv1D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
FILE_NAME = 'Final_Model_Data.csv'
TEST_SPLIT = 0.2
TIME_STEPS = 24  # Look back at the last 24 hours to predict the next 1

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
print("⏳ Loading Final Dataset for Sliding Window Model...")
df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)

feature_cols = [c for c in df.columns if c != 'pm25']
target_col = 'pm25'

print(f"📊 Features used: {feature_cols}")

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# ==============================================================================
# 3. SPLIT TRAIN vs TEST
# ==============================================================================
split_idx = int(len(df) * (1 - TEST_SPLIT))

X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

# ==============================================================================
# 4. NORMALIZATION
# ==============================================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw)

X_test_scaled = scaler_X.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# ==============================================================================
# 5. SLIDING WINDOW (CREATE SEQUENCES)
# ==============================================================================
def create_sequences(X_data, y_data, time_steps):
    Xs, ys = [], []
    # Loop to create sequences
    for i in range(len(X_data) - time_steps):
        # Take a window of 'time_steps' rows
        Xs.append(X_data[i:(i + time_steps)])
        # Take the target value immediately after the window
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)

print(f"🔄 Applying Sliding Window (TimeSteps={TIME_STEPS})...")

X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, TIME_STEPS)

print(f"   Original Train Rows: {len(X_train_scaled)}")
print(f"   Sequenced Train Shape: {X_train.shape}  <-- (Samples, TimeSteps, Features)")
print(f"   Sequenced Test Shape:  {X_test.shape}")

# ==============================================================================
# 6. BUILD MODEL (CNN-BiLSTM + ATTENTION)
# ==============================================================================
print("🧠 Building Advanced Hybrid Network...")

# Input Shape is now (TIME_STEPS, Features)
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# --- 1. CNN Layer ---
# Increased kernel_size to 3 to find local patterns in the 24-hour window
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)

# --- 2. Bi-LSTM Layer ---
lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
lstm_out = Dropout(0.2)(lstm_out)

# --- 3. Attention Mechanism ---
# Calculate scores
attention_score = Dense(128, activation='tanh')(lstm_out)
attention_weights = Dense(1, activation='softmax')(attention_score)

# Apply weights (Context Vector)
context_vector = Multiply()([lstm_out, attention_weights])
context_vector = Flatten()(context_vector)

# --- 4. Output Layers ---
x = Dense(64, activation='relu')(context_vector)
x = Dropout(0.2)(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# ==============================================================================
# 7. TRAIN
# ==============================================================================
print("🚀 Training Started...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ==============================================================================
# 8. PREDICT & VISUALIZE
# ==============================================================================
print("🔮 Forecasting...")
predictions_scaled = model.predict(X_test)

predictions_actual = scaler_y.inverse_transform(predictions_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

plt.figure(figsize=(15, 6))
limit = 500
plt.plot(y_test_actual[:limit], label='Actual PM2.5', color='blue', alpha=0.6)
plt.plot(predictions_actual[:limit], label='Prediction', color='red', linestyle='--')
plt.title(f'Final Model (Window={TIME_STEPS}): Actual vs Predicted')
plt.xlabel('Hours')
plt.ylabel('PM2.5 Level')
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# 9. METRICS
# ==============================================================================
print("📊 Calculating Final Metrics...")

# Train Metrics
train_pred_scaled = model.predict(X_train)
y_train_act = scaler_y.inverse_transform(y_train)
train_pred_act = scaler_y.inverse_transform(train_pred_scaled)

rmse_train = math.sqrt(mean_squared_error(y_train_act, train_pred_act))
mae_train = mean_absolute_error(y_train_act, train_pred_act)
r2_train = r2_score(y_train_act, train_pred_act)

# Test Metrics
rmse_test = math.sqrt(mean_squared_error(y_test_actual, predictions_actual))
mae_test = mean_absolute_error(y_test_actual, predictions_actual)
r2_test = r2_score(y_test_actual, predictions_actual)

print("-" * 60)
print(f"{'METRIC':<10} | {'TRAIN':<18} | {'TEST':<18}")
print("-" * 60)
print(f"{'RMSE':<10} | {rmse_train:<18.2f} | {rmse_test:<18.2f}")
print(f"{'MAE':<10} | {mae_train:<18.2f} | {mae_test:<18.2f}")
print(f"{'R²':<10} | {r2_train:<18.4f} | {r2_test:<18.4f}")
print("-" * 60)

# ==============================================================================
# 10. SAVE
# ==============================================================================
print("💾 Saving Model...")
model.save('best_model_window24.keras')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✅ Saved.")
