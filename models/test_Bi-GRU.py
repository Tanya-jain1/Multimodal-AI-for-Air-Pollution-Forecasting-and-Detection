import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Bidirectional # <--- Combined here
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
FILE_NAME = 'Final_Model_Data.csv'
TEST_SPLIT = 0.2  # Use last 20% of data for testing

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
print("⏳ Loading Final Dataset for Bi-GRU...")
df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)

# Select all columns EXCEPT the target (pm25) as features
feature_cols = [c for c in df.columns if c != 'pm25']
target_col = 'pm25'

print(f"📊 Features used: {feature_cols}")

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# ==============================================================================
# 3. SPLIT TRAIN vs TEST (Strictly by Time)
# ==============================================================================
split_idx = int(len(df) * (1 - TEST_SPLIT))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"🚂 Training Rows: {len(X_train)}")
print(f"🧪 Testing Rows:  {len(X_test)}")

# ==============================================================================
# 4. NORMALIZATION (Scaling 0 to 1)
# ==============================================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Apply the same math to the test set
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# ==============================================================================
# 5. RESHAPE FOR Bi-GRU
# ==============================================================================
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ==============================================================================
# 6. BUILD MODEL (Bi-GRU VERSION)
# ==============================================================================
print("🧠 Building Bidirectional GRU Neural Network...")
model = Sequential()

# Input Layer
model.add(Input(shape=(1, X_train.shape[1])))

# Bi-GRU Layer 1
# We wrap the GRU layer inside Bidirectional()
model.add(Bidirectional(GRU(64, return_sequences=True))) 
model.add(Dropout(0.2))

# Bi-GRU Layer 2
model.add(Bidirectional(GRU(32, return_sequences=False)))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# ==============================================================================
# 7. TRAIN
# ==============================================================================
print("🚀 Training Bi-GRU Started...")
history = model.fit(
    X_train_reshaped, y_train_scaled,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test_scaled),
    verbose=1
)

# ==============================================================================
# 8. PREDICT & VISUALIZE
# ==============================================================================
print("🔮 Forecasting with Bi-GRU...")
predictions_scaled = model.predict(X_test_reshaped)

# Convert predictions back to AQI values
predictions_actual = scaler_y.inverse_transform(predictions_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# PLOT RESULTS
plt.figure(figsize=(15, 6))
limit = 500
plt.plot(y_test_actual[:limit], label='Actual PM2.5', color='blue', alpha=0.6)
plt.plot(predictions_actual[:limit], label='Bi-GRU Prediction', color='orange', linestyle='--') # Orange for Bi-GRU
plt.title('Final Bi-GRU Model: Actual vs Predicted')
plt.xlabel('Hours')
plt.ylabel('PM2.5 Level')
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# 9. CALCULATE METRICS (TRAINING SET)
# ==============================================================================
print("📊 Calculating Bi-GRU Training Metrics...")

train_predict_scaled = model.predict(X_train_reshaped)
y_train_actual = scaler_y.inverse_transform(y_train_scaled)
train_predict_actual = scaler_y.inverse_transform(train_predict_scaled)

mse_train = mean_squared_error(y_train_actual, train_predict_actual)
rmse_train = math.sqrt(mse_train)
mae_train = mean_absolute_error(y_train_actual, train_predict_actual)
r2_train = r2_score(y_train_actual, train_predict_actual)

print("-" * 40)
print(f"🏆 Bi-GRU TRAINING PERFORMANCE:")
print(f"   - RMSE: {rmse_train:.2f}")
print(f"   - MAE:  {mae_train:.2f}")
print(f"   - R²:   {r2_train:.4f}")
print("-" * 40)

# ==============================================================================
# 10. CALCULATE & COMPARE TEST METRICS
# ==============================================================================
print("\n📊 Calculating Bi-GRU TEST Metrics...")

test_predictions_scaled = model.predict(X_test_reshaped)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)
test_predictions_actual = scaler_y.inverse_transform(test_predictions_scaled)

mse_test = mean_squared_error(y_test_actual, test_predictions_actual)
rmse_test = math.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_actual, test_predictions_actual)
r2_test = r2_score(y_test_actual, test_predictions_actual)

print("-" * 60)
print(f"{'METRIC':<10} | {'TRAIN':<18} | {'TEST':<18}")
print("-" * 60)
print(f"{'RMSE':<10} | {rmse_train:<18.2f} | {rmse_test:<18.2f}")
print(f"{'MAE':<10} | {mae_train:<18.2f} | {mae_test:<18.2f}")
print(f"{'R²':<10} | {r2_train:<18.4f} | {r2_test:<18.4f}")
print("-" * 60)

# ==============================================================================
# 11. SAVE EVERYTHING
# ==============================================================================
print("💾 Saving Bi-GRU Model...")

# 1. Save the AI Brain
model.save('best_bigru_model.keras') # <--- Changed Name
print("✅ Model saved as 'best_bigru_model.keras'")

# 2. Save Scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✅ Scalers saved/updated.")