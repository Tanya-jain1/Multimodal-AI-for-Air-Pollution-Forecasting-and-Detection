import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
MODEL_PATH = 'best_cnn_bilstm_model.keras'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
FEATURES_PATH = 'model_features.pkl'

# ==============================================================================
# 2. LOAD ARTIFACTS
# ==============================================================================
print("⏳ Loading system artifacts...")

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Load the scalers
try:
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    print("✅ Scalers loaded successfully.")
except Exception as e:
    print(f"❌ Error loading scalers: {e}")
    exit()

# Load the feature list
try:
    with open(FEATURES_PATH, 'rb') as f:
        feature_cols = pickle.load(f)
    print(f"✅ Feature list loaded. Expecting {len(feature_cols)} features.")
except Exception as e:
    print(f"❌ Error loading feature list: {e}")
    exit()

# ==============================================================================
# 3. SIMULATE LIVE DATA INPUT
# ==============================================================================
# In a real scenario, this dictionary comes from an API request.
# For now, we manually create a "dummy" row of data to test the pipeline.
# IMPORTANT: These keys must match the column names in your original CSV exactly.

live_data_sample = {
    'CO': 0.8,
    'NO': 15.2,
    'NO2': 22.5,
    'O3': 18.0,
    'SO2': 5.5,
    'PM10': 95.0,
    'NH3': 4.2,
    'Temperature': 28.5,
    'Humidity': 65.0,
    'WindSpeed': 3.5,
    # Add any other columns your model was trained on here
}

print("\n📡 Received live data input...")

# ==============================================================================
# 4. PREPROCESS DATA
# ==============================================================================

# Step A: Convert to DataFrame
input_df = pd.DataFrame([live_data_sample])

# Step B: Align Columns (CRITICAL)
# This ensures we have the exact columns in the exact order as training.
# If a column is missing in live data, it fills with 0 (safety net).
input_df = input_df.reindex(columns=feature_cols, fill_value=0)

# Step C: Scale the Features
# We use .transform(), NOT .fit_transform()
input_scaled = scaler_X.transform(input_df)

# Step D: Reshape for CNN-BiLSTM
# The model expects: (Samples, TimeSteps, Features) -> (1, 1, N_Features)
input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))

# ==============================================================================
# 5. PREDICT
# ==============================================================================
print("🧠 Analyzing air quality...")

# Get the scaled prediction (0 to 1)
prediction_scaled = model.predict(input_reshaped, verbose=0)

# Convert back to real AQI (PM2.5) value
prediction_actual = scaler_y.inverse_transform(prediction_scaled)
predicted_pm25 = prediction_actual[0][0]

# ==============================================================================
# 6. RESULT
# ==============================================================================
print("-" * 30)
print(f"🔮 FORECAST RESULT")
print("-" * 30)
print(f"Predicted PM2.5 Level: {predicted_pm25:.2f}")

# Optional: Add a simple health warning based on the result
if predicted_pm25 <= 50:
    print("Health Status: 🟢 Good")
elif predicted_pm25 <= 100:
    print("Health Status: 🟡 Moderate")
elif predicted_pm25 <= 200:
    print("Health Status: 🟠 Unhealthy")
else:
    print("Health Status: 🔴 Hazardous")
print("-" * 30)