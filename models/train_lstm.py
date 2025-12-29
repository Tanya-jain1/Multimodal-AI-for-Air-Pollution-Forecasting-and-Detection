import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
FILE_NAME = 'Final_Model_Data.csv'
TEST_SPLIT = 0.2  # Use last 20% of data for testing (approx 2024-2025)

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
print("⏳ Loading Final Dataset...")
df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)

# Select all columns EXCEPT the target (pm25) as features
feature_cols = [c for c in df.columns if c != 'pm25']
target_col = 'pm25'

print(f"📊 Features used: {feature_cols}")

# Create X (Features) and y (Target)
X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# ==============================================================================
# 3. SPLIT TRAIN vs TEST (Strictly by Time)
# ==============================================================================
# We cannot shuffle time-series data!
split_idx = int(len(df) * (1 - TEST_SPLIT))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"🚂 Training Rows: {len(X_train)}")
print(f"🧪 Testing Rows:  {len(X_test)}")

# ==============================================================================
# 4. NORMALIZATION (Scaling 0 to 1)
# ==============================================================================
# We fit the scaler ONLY on training data to avoid "cheating" (Data Leakage)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Apply the same math to the test set
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# ==============================================================================
# 5. RESHAPE FOR LSTM
# ==============================================================================
# LSTM needs 3D input: (Samples, TimeSteps, Features)
# Since we made Lag columns manually, TimeSteps = 1
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ==============================================================================
# 6. BUILD MODEL
# ==============================================================================
print("🧠 Building LSTM Neural Network...")
model = Sequential()

# Input Layer + LSTM Layer 1
model.add(Input(shape=(1, X_train.shape[1])))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2)) # Prevents overfitting

# LSTM Layer 2
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))

# Output Layer (Predict 1 value)
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# ==============================================================================
# 7. TRAIN
# ==============================================================================
print("🚀 Training Started... (This may take 1-2 minutes)")
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
print("🔮 Forecasting...")
predictions_scaled = model.predict(X_test_reshaped)

# Convert "0.5" back to "150 AQI"
predictions_actual = scaler_y.inverse_transform(predictions_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# PLOT RESULTS
plt.figure(figsize=(15, 6))
# We plot a slice of 500 hours to verify if the lines match
limit = 500
plt.plot(y_test_actual[:limit], label='Actual PM2.5', color='blue', alpha=0.6)
plt.plot(predictions_actual[:limit], label='AI Prediction', color='red', linestyle='--')
plt.title('Final LSTM Model: Actual vs Predicted (First 500 Hours of Test Set)')
plt.xlabel('Hours')
plt.ylabel('PM2.5 Level')
plt.legend()
plt.grid(True)
plt.show()

print("✅ DONE! Check the graph.")


# ==============================================================================
# 9. CALCULATE METRICS (TRAINING SET)
# ==============================================================================
print("📊 Calculating Training Metrics...")

# 1. Make Predictions on the Training Set
train_predict_scaled = model.predict(X_train_reshaped)

# 2. Inverse Scale (Convert "0.5" back to "150 AQI" so metrics make sense)
# We must use the same scaler we used for 'y'
y_train_actual = scaler_y.inverse_transform(y_train_scaled)
train_predict_actual = scaler_y.inverse_transform(train_predict_scaled)

# 3. Calculate the Scores
# MSE (Mean Squared Error)
mse_train = mean_squared_error(y_train_actual, train_predict_actual)

# RMSE (Root Mean Squared Error)
rmse_train = math.sqrt(mse_train)

# MAE (Mean Absolute Error)
mae_train = mean_absolute_error(y_train_actual, train_predict_actual)

# R2 Score (Coefficient of Determination)
r2_train = r2_score(y_train_actual, train_predict_actual)

# 4. Print Results
print("-" * 40)
print(f"🏆 TRAINING PERFORMANCE METRICS:")
print(f"   - RMSE: {rmse_train:.2f}  (Lower is better)")
print(f"   - MAE:  {mae_train:.2f}   (Lower is better)")
print(f"   - MSE:  {mse_train:.2f}   (Lower is better)")
print(f"   - R²:   {r2_train:.4f}   (Closer to 1.0 is better)")
print("-" * 40)

# ==============================================================================
# 10. CALCULATE & COMPARE TEST METRICS
# ==============================================================================
print("\n📊 Calculating TEST Metrics (The Real Exam)...")

# 1. Make Predictions on Test Data (You likely already did this step)
test_predictions_scaled = model.predict(X_test_reshaped)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)
test_predictions_actual = scaler_y.inverse_transform(test_predictions_scaled)

# 2. Calculate Test Scores
mse_test = mean_squared_error(y_test_actual, test_predictions_actual)
rmse_test = math.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_actual, test_predictions_actual)
r2_test = r2_score(y_test_actual, test_predictions_actual)

# 3. Print Comparison Table
print("-" * 60)
print(f"{'METRIC':<10} | {'TRAIN (Learned)':<18} | {'TEST (New Data)':<18}")
print("-" * 60)
print(f"{'RMSE':<10} | {rmse_train:<18.2f} | {rmse_test:<18.2f}")
print(f"{'MAE':<10} | {mae_train:<18.2f} | {mae_test:<18.2f}")
print(f"{'R²':<10} | {r2_train:<18.4f} | {r2_test:<18.4f}")
print("-" * 60)

if r2_test > 0.80:
    print("✅ SUCCESS: Your model generalizes well to new future data!")
else:
    print("⚠️ WARNING: Significant drop in performance. Check for overfitting.")


# ==============================================================================
# 11. SAVE EVERYTHING
# ==============================================================================
print("💾 Saving your hard work...")

# 1. Save the AI Brain (Keras format is best for TensorFlow)
model.save('best_lstm_model.keras')
print("✅ Model saved as 'best_lstm_model.keras'")

# 2. Save the Translators (Scalers)
# We use joblib (standard for saving Scikit-Learn tools)
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✅ Scalers saved as 'scaler_X.pkl' and 'scaler_y.pkl'")

print("\nSAFE! You can now close VS Code or crash your computer without worry.")