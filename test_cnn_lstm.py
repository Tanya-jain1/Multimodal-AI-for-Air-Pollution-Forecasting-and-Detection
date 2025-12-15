
#CNN LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import platform
import subprocess

# ===================================================================
# Configuration
# ===================================================================
AQ_DATA_FILE = "delhi_aq_featured_daily.csv"
WEATHER_DATA_FILE = "delhi_weather_processed.csv"
TARGET_VARIABLE = "pm25"
TIME_STEPS = 10  # How many past days of data the model will see
EPOCHS = 50
BATCH_SIZE = 32

# ===================================================================
# 1. Load and Merge Data
# ===================================================================
print("🚀 Step 1: Loading and merging data...")
# Note: Ensure your data files are in the same directory as the script
# or provide the full path to them.
try:
    aq_df = pd.read_csv(AQ_DATA_FILE, parse_dates=True, index_col='datetime')
    weather_df = pd.read_csv(WEATHER_DATA_FILE, parse_dates=True, index_col='DATE')
    df_merged = aq_df.join(weather_df, how='inner')
    print("✅ Data merged successfully. Shape:", df_merged.shape)
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Make sure the data files are in the correct location.")
    exit()

# ===================================================================
# 2. Prepare Data for CNN-LSTM
# ===================================================================
print("\n🚀 Step 2: Preparing data for the model...")

# Handle non-numeric columns by dropping them
object_cols = df_merged.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"Found non-numeric columns: {list(object_cols)}. Dropping them.")
    df_merged = df_merged.drop(columns=object_cols)
    
# Select features (X) and target (y)
features = df_merged.drop(columns=[TARGET_VARIABLE])
target = df_merged[[TARGET_VARIABLE]]

# Scale the data
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(features)
y_scaled = target_scaler.fit_transform(target)

# Create sequences
def create_sequences(X, y, time_steps=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
print(f"✅ Data preparation complete. X_train shape: {X_train.shape}")

# ===================================================================
# 3. Build the Hybrid CNN-LSTM Model
# ===================================================================
print("\n🚀 Step 3: Building the Hybrid CNN-LSTM model...")

input_shape = (X_train.shape[1], X_train.shape[2]) # (TIME_STEPS, num_features)
input_layer = Input(shape=input_shape)

# --- CNN Feature Extractor Part ---
conv_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)

# --- LSTM Sequence Modeler Part ---
lstm_layer = LSTM(100, activation='relu')(pooling_layer)
output_layer = Dense(1, activation='linear')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
model.summary()

# ===================================================================
# 4. Train the Model
# ===================================================================
print("\n🚀 Step 4: Training the CNN-LSTM model...")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

print("✅ Model training complete.")

# ===================================================================
# 5. Evaluate the Model
# ===================================================================
print("\n🚀 Step 5: Evaluating the model performance...")

y_pred_scaled = model.predict(X_test)

# Inverse transform to get actual values
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_actual = target_scaler.inverse_transform(y_test)

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("---------------------------------")

# ===================================================================
# 6. Visualize the Results and Open the File
# ===================================================================
print("\n🚀 Step 6: Visualizing and opening the results plot...")

plt.figure(figsize=(15, 6))
plt.plot(y_test_actual, label='Actual PM2.5', color='blue')
plt.plot(y_pred, label='Predicted PM2.5', color='green', linestyle='--')
plt.title('Hybrid CNN-LSTM Model: Actual vs. Predicted PM2.5')
plt.xlabel('Time')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.grid(True)

# Save the plot
output_filename = "cnn_lstm_prediction_plot.png"
plt.savefig(output_filename)
print(f"✅ Plot saved as '{output_filename}'")

# --- MODIFICATION: Open the saved plot automatically ---
try:
    current_os = platform.system()
    if current_os == 'Windows':
        os.startfile(output_filename)
    elif current_os == 'Darwin': # macOS
        subprocess.call(['open', output_filename])
    elif current_os == 'Linux':
        subprocess.call(['xdg-open', output_filename])
    else:
        print(f"Unsupported OS '{current_os}'. Please open the file manually: {output_filename}")
except Exception as e:
    print(f"Error opening file: {e}. Please open it manually: {output_filename}")
# --- END MODIFICATION ---

print("\n✅ All steps complete!")