import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# ==============================================================================
# 1. CONFIGURATION & 2. LOAD DATA
# ==============================================================================
FILE_NAME = 'Final_Model_Data.csv'
TEST_SPLIT = 0.2 

print("⏳ Loading Final Dataset for CNN...")
df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)

feature_cols = [c for c in df.columns if c != 'pm25']
target_col = 'pm25'

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# ==============================================================================
# 3. SPLIT & 4. NORMALIZATION
# ==============================================================================
split_idx = int(len(df) * (1 - TEST_SPLIT))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# ==============================================================================
# 5. RESHAPE FOR CNN (1D)
# ==============================================================================
# CNN 1D expects: (Samples, TimeSteps, Features)
# Since we are using a "Point-in-Time" prediction, TimeSteps = 1
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ==============================================================================
# 6. BUILD MODEL (CNN VERSION)
# ==============================================================================
print("🧠 Building 1D CNN Neural Network...")
model = Sequential()

# CNN Layer 1: Filters look for patterns across features
model.add(Input(shape=(1, X_train.shape[1])))
model.add(Conv1D(filters=64, kernel_size=1, activation='relu')) 
model.add(Dropout(0.2))

# CNN Layer 2
model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
model.add(Dropout(0.2))

# Flattening the 3D output to 2D for the Dense layers
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(50, activation='relu'))
model.add(Dense(1)) # Output Layer

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# ==============================================================================
# 7. TRAIN
# ==============================================================================
print("🚀 Training CNN Started...")
history = model.fit(
    X_train_reshaped, y_train_scaled,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test_scaled),
    verbose=1
)

# ==============================================================================
# 8. PREDICT & VISUALIZE (Simplified for brevity)
# ==============================================================================
print("🔮 Forecasting with CNN...")
predictions_scaled = model.predict(X_test_reshaped)
predictions_actual = scaler_y.inverse_transform(predictions_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

plt.figure(figsize=(15, 6))
plt.plot(y_test_actual[:500], label='Actual PM2.5', color='blue', alpha=0.6)
plt.plot(predictions_actual[:500], label='CNN Prediction', color='red', linestyle='--')
plt.title('Final CNN Model: Actual vs Predicted')
plt.legend()
plt.show()
# ... (Previous code for data loading and model training remains the same)

# ==============================================================================
# 9. CALCULATE METRICS (TRAINING SET)
# ==============================================================================
print("📊 Calculating CNN Training Metrics...")

train_predict_scaled = model.predict(X_train_reshaped)
y_train_actual = scaler_y.inverse_transform(y_train_scaled)
train_predict_actual = scaler_y.inverse_transform(train_predict_scaled)

mse_train = mean_squared_error(y_train_actual, train_predict_actual)
rmse_train = math.sqrt(mse_train)
mae_train = mean_absolute_error(y_train_actual, train_predict_actual) # <--- Added MAE
r2_train = r2_score(y_train_actual, train_predict_actual)

# ==============================================================================
# 10. CALCULATE & COMPARE TEST METRICS
# ==============================================================================
print("\n📊 Calculating CNN TEST Metrics...")

test_predictions_scaled = model.predict(X_test_reshaped)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)
test_predictions_actual = scaler_y.inverse_transform(test_predictions_scaled)

mse_test = mean_squared_error(y_test_actual, test_predictions_actual)
rmse_test = math.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_actual, test_predictions_actual) # <--- Added MAE
r2_test = r2_score(y_test_actual, test_predictions_actual)

print("-" * 60)
print(f"{'METRIC':<10} | {'TRAIN':<18} | {'TEST':<18}")
print("-" * 60)
print(f"{'RMSE':<10} | {rmse_train:<18.2f} | {rmse_test:<18.2f}")
print(f"{'MAE':<10} | {mae_train:<18.2f} | {mae_test:<18.2f}") # <--- Displayed MAE
print(f"{'R²':<10} | {r2_train:<18.4f} | {r2_test:<18.4f}")
print("-" * 60)

# ==============================================================================
# 11. SAVE
# ==============================================================================
model.save('best_cnn_model.keras')
print("✅ CNN Model and Scalers saved.")