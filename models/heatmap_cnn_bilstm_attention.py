import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Conv1D
from tensorflow.keras.optimizers import Adam
import math

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
FILE_NAME = 'Final_Model_Data.csv'
TEST_SPLIT = 0.2

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
print("⏳ Loading Data...")
df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)

feature_cols = [c for c in df.columns if c != 'pm25']
target_col = 'pm25'
X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# ==============================================================================
# 3. SPLIT & SCALE (Must match training exactly)
# ==============================================================================
split_idx = int(len(df) * (1 - TEST_SPLIT))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# Reshape for CNN-BiLSTM (Samples, 1, Features)
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ==============================================================================
# 4. LOAD THE CHAMPION MODEL
# ==============================================================================
print("🧠 Loading the Champion Model (CNN-BiLSTM)...")
# Make sure this matches the filename you saved earlier!
model = load_model('best_cnn_bilstm_model.keras') 
print("✅ Model Loaded Successfully.")

# ==============================================================================
# 5. GENERATE SALIENCY HEATMAP 🌡️
# ==============================================================================
print("\n🔍 Generating Feature Importance Heatmap...")

# 1. Select 20 random examples from the test set to visualize
sample_size = 20
indices = np.random.choice(len(X_test_reshaped), sample_size, replace=False)
sample_input = X_test_reshaped[indices]

# 2. Convert to Tensor
sample_input_tensor = tf.convert_to_tensor(sample_input, dtype=tf.float32)

# 3. Calculate Gradients
# We ask: "How much does the Output change if we change this specific Input Feature?"
with tf.GradientTape() as tape:
    tape.watch(sample_input_tensor)
    predictions = model(sample_input_tensor)

# Get the gradients
grads = tape.gradient(predictions, sample_input_tensor)

# 4. Process Gradients
# Take absolute value (we care about magnitude, not direction)
# Shape is (20, 1, 10). We remove the middle dimension.
saliency = tf.reduce_mean(tf.abs(grads), axis=1).numpy()

# 5. Create DataFrame for Plotting
heatmap_df = pd.DataFrame(saliency, columns=feature_cols)

# 6. Plot
plt.figure(figsize=(12, 6))
# Transpose (.T) so Features are on the Left (Y-axis)
sns.heatmap(heatmap_df.T, cmap='inferno', annot=False) 

plt.title(f'Why did the AI predict this? (Feature Saliency Map)\nModel: CNN-BiLSTM (R²=0.9449)')
plt.xlabel('Random Test Samples (Hours)')
plt.ylabel('Input Features')
plt.tight_layout()
plt.show()

print("✅ Heatmap Generated!")