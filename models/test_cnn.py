import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# ==============================================================================
# 1. SETUP
# ==============================================================================
FILE_NAME = 'Final_Model_Data.csv'
TIME_STEPS = 24  # CNN needs a window to see patterns

# ==============================================================================
# 2. DATA PREP (Same as before)
# ==============================================================================
df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)
feature_cols = [c for c in df.columns if c != 'pm25']
target_col = 'pm25'
X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

split_idx = int(len(df) * 0.8)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_raw = scaler_X.fit_transform(X[:split_idx])
y_train_raw = scaler_y.fit_transform(y[:split_idx])
X_test_raw = scaler_X.transform(X[split_idx:])
y_test_raw = scaler_y.transform(y[split_idx:])

def create_sequences(X_data, y_data, time_steps):
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        Xs.append(X_data[i:(i + time_steps)])
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(X_train_raw, y_train_raw, TIME_STEPS)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, TIME_STEPS)

# ==============================================================================
# 3. BUILD STANDARD CNN
# ==============================================================================
print("🧠 Building Standard CNN...")
model = Sequential()

model.add(Input(shape=(TIME_STEPS, X_train.shape[2])))

# Layer 1: Conv1D (Scans the 24-hour window)
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2)) # Shrinks data, keeps strongest features

# Layer 2: Conv1D (Finds higher-level patterns)
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
# No pooling here to keep some detail

# Flatten: Convert 3D maps to 2D for the dense layer
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# ==============================================================================
# 4. TRAIN & EVALUATE
# ==============================================================================
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_data=(X_test, y_test))

print("\n📊 Calculating Standard CNN Metrics...")
test_predictions = model.predict(X_test)
y_test_inv = scaler_y.inverse_transform(y_test)
pred_inv = scaler_y.inverse_transform(test_predictions)

rmse = math.sqrt(mean_squared_error(y_test_inv, pred_inv))
mae = mean_absolute_error(y_test_inv, pred_inv)
r2 = r2_score(y_test_inv, pred_inv)

print("-" * 40)
print(f"STANDARD CNN RESULTS:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.4f}")
print("-" * 40)