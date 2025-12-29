import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
FILE_NAME = 'Final_Model_Data.csv'  # Ensure this matches your file name
MODEL_NAME = 'best_cnn_bilstm_model.keras'

print("⏳ Loading Data...")
df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)

# --- RE-CREATE FEATURE_COLS ---
# We define features as "Everything except pm25"
feature_cols = [c for c in df.columns if c != 'pm25']
target_col = 'pm25'
print(f"✅ Features Identified: {feature_cols}")

# Prepare Data
X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# ==============================================================================
# 2. PREPROCESSING (Must match training exactly)
# ==============================================================================
# Split Train/Test (Last 20% is test)
TEST_SPLIT = 0.2
split_idx = int(len(df) * (1 - TEST_SPLIT))
X_test = X[split_idx:]

# Scale Data
scaler_X = MinMaxScaler()
# Note: In a real scenario, you should load the saved scaler. 
# For now, we fit on the full X to approximate the scaling if the pkl isn't handy.
scaler_X.fit(X) 
X_test_scaled = scaler_X.transform(X_test)

# Reshape for Model: (Samples, 1, Features)
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ==============================================================================
# 3. LOAD MODEL
# ==============================================================================
print(f"🧠 Loading Model: {MODEL_NAME}...")
model = tf.keras.models.load_model(MODEL_NAME)

# ==============================================================================
# 4. COMPUTE SALIENCY
# ==============================================================================
def compute_saliency_map(model, input_data):
    """
    Computes gradients of the output w.r.t input features.
    """
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
    
    # Calculate gradients
    gradients = tape.gradient(predictions, input_tensor)
    
    # Take absolute value (magnitude)
    saliency = tf.abs(gradients)
    
    # Remove the time dimension (Batch, 1, Features) -> (Batch, Features)
    saliency = tf.squeeze(saliency, axis=1)
    
    return saliency.numpy()

# Select 20 Random Samples for the Plot
NUM_SAMPLES = 20
indices = np.random.choice(len(X_test_reshaped), NUM_SAMPLES, replace=False)
sample_inputs = X_test_reshaped[indices]

print("🔍 Calculating Gradients...")
saliency_values = compute_saliency_map(model, sample_inputs)

# Transpose for plotting: Features on Y-axis, Samples on X-axis
saliency_map_plotting = saliency_values.T

# ==============================================================================
# 5. PLOT IEEE FIGURE
# ==============================================================================
plt.figure(figsize=(12, 6))

ax = sns.heatmap(
    saliency_map_plotting, 
    yticklabels=feature_cols,  # Uses the list we re-created above
    xticklabels=indices,       # Shows the original index of the sample
    cmap='inferno',            # 'inferno' or 'viridis' look professional
    cbar_kws={'label': 'Gradient Magnitude (Importance)'}
)

plt.title('Fig. 4. Feature Saliency Map (CNN-BiLSTM Gradient Analysis)', fontsize=14)
plt.xlabel('Random Test Sample Indices', fontsize=12)
plt.ylabel('Input Features', fontsize=12)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('Fig4_Saliency_Map.png', dpi=300)
print("✅ Saved 'Fig4_Saliency_Map.png'")
plt.show()