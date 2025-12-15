#GRU
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ===================================================================
# Configuration
# ===================================================================
AQ_DATA_FILE = "delhi_aq_featured_daily.csv"
WEATHER_DATA_FILE = "delhi_weather_processed.csv"
TARGET_VARIABLE = "pm25"
TIME_STEPS = 10  # How many past days of data the model will see to predict the next day
EPOCHS = 50
BATCH_SIZE = 32

# ===================================================================
# 1. Load and Merge Data
# ===================================================================
print("🚀 Step 1: Loading and merging data...")

# Load the datasets
aq_df = pd.read_csv(AQ_DATA_FILE, parse_dates=True, index_col='datetime')
weather_df = pd.read_csv(WEATHER_DATA_FILE, parse_dates=True, index_col='DATE')

# Merge the two dataframes on their date index
df_merged = aq_df.join(weather_df, how='inner')

print("✅ Data merged successfully. Shape:", df_merged.shape)
print("Merged data preview:\n", df_merged.head())

# ===================================================================
# 2. Prepare Data for GRU
# ===================================================================
print("\n🚀 Step 2: Preparing data for the GRU model...")

# --- FIX: Handle non-numeric columns ---
# First, let's identify any columns that are not numbers.
object_cols = df_merged.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"Found non-numeric columns: {list(object_cols)}. Dropping them.")
    df_merged = df_merged.drop(columns=object_cols)
    # If you wanted to do one-hot encoding instead (the better fix), you would use:
    # df_merged = pd.get_dummies(df_merged, columns=object_cols, drop_first=True)
else:
    print("No non-numeric columns found. Proceeding.")
    
# Select features (X) and target (y)
features = df_merged.drop(columns=[TARGET_VARIABLE])
target = df_merged[[TARGET_VARIABLE]]

# Scale the data for better model performance
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(features)
y_scaled = target_scaler.fit_transform(target)

# --- Function to create sequences ---
def create_sequences(X, y, time_steps=10):
    """Creates sequences of data for time-series forecasting."""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# Create the sequences
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)
print(f"✅ Data sequenced. X shape: {X_seq.shape}, y shape: {y_seq.shape}")

# Split data into training and testing sets (80% train, 20% test)
# For time-series, we don't shuffle the data to maintain chronological order
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
print(f"Data split into training and testing sets.")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# ===================================================================
# 3. Build the GRU Model
# ===================================================================
print("\n🚀 Step 3: Building the GRU model...")

# Get the input shape for the model
input_shape = (X_train.shape[1], X_train.shape[2]) # (TIME_STEPS, num_features)

# Define the model architecture using Keras Functional API
input_layer = Input(shape=input_shape)
# A GRU layer with 128 units. You can experiment with this number.
gru_layer = GRU(128, activation='relu')(input_layer)
# Output layer with a single neuron for our single target value
output_layer = Dense(1, activation='linear')(gru_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
model.summary()

# ===================================================================
# 4. Train the Model
# ===================================================================
print("\n🚀 Step 4: Training the GRU model...")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1, # Use 10% of training data for validation
    verbose=1
)

print("✅ Model training complete.")

# ===================================================================
# 5. Evaluate the Model
# ===================================================================
print("\n🚀 Step 5: Evaluating the model performance...")

# Make predictions on the test set
y_pred_scaled = model.predict(X_test)

# **IMPORTANT**: Inverse transform the predictions and actual values
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_actual = target_scaler.inverse_transform(y_test)

# --- Here is the calculation ---
mse = mean_squared_error(y_test_actual, y_pred) # Mean Squared Error
rmse = np.sqrt(mse)                             # Root Mean Squared Error
mae = mean_absolute_error(y_test_actual, y_pred) # Mean Absolute Error
r2 = r2_score(y_test_actual, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("---------------------------------")

# ===================================================================
# 6. Visualize the Results
# ===================================================================
# ADD THIS IMPORT AT THE TOP OF YOUR SCRIPT
import os 
# If you've already imported it, you don't need to do it again.
# An alternative that works on all operating systems is: import webbrowser
# ===================================================================

print("\n🚀 Step 6: Visualizing the results...")

# ... (The first part of Step 6 remains the same) ...
#
# (Keep your existing plt.figure, plt.plot, plt.title, etc. code here)
plt.figure(figsize=(15, 6))
plt.plot(y_test_actual, label='Actual PM2.5', color='blue')
plt.plot(y_pred, label='Predicted PM2.5', color='red', linestyle='--')
plt.title('GRU Model: Actual vs. Predicted PM2.5')
plt.xlabel('Time')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.grid(True)
#
# ... (End of the plotting code) ...

# Save the plot to a file
output_filename = "gru_prediction_plot.png"
plt.savefig(output_filename)
print(f"✅ Plot saved as '{output_filename}'")

# Automatically open the saved image file
try:
    os.startfile(output_filename) # For Windows
    # For macOS, use: os.system(f"open {output_filename}")
    # For Linux, use: os.system(f"xdg-open {output_filename}")
    # Or use the cross-platform version: webbrowser.open(output_filename)
except AttributeError:
    print("\nCould not automatically open the file. Please open it manually.")

print("✅ All steps complete!")
