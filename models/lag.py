import numpy as np
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import math

# ==============================================================================
# 1. CONFIGURATION & LOADING
# ==============================================================================
MODEL_PATH = 'best_cnn_bilstm_model.keras'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
FEATURES_PATH = 'model_features.pkl'
DATA_FILE = 'Final_Model_Data.csv' # Your test data source

print("⏳ Loading Model & Artifacts...")
model = load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)
with open(FEATURES_PATH, 'rb') as f:
    feature_cols = pickle.load(f)

# ==============================================================================
# 2. PREPARE TEST DATA (Exact same way as Training)
# ==============================================================================
print("📊 Preparing Test Data...")
df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

# Split last 20% for testing (Must match your training logic)
split_idx = int(len(df) * 0.8) 
df_test = df.iloc[split_idx:]

X_test = df_test[feature_cols].values
y_test = df_test['pm25'].values

# Scale
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Reshape for CNN-BiLSTM (Samples, TimeSteps, Features)
# Assuming TimeSteps=1 based on your previous code
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ==============================================================================
# 3. CALCULATE BASELINE PERFORMANCE
# ==============================================================================
print("📉 Calculating Baseline Error...")
preds_baseline = model.predict(X_test_reshaped, verbose=0)
mse_baseline = mean_squared_error(y_test_scaled, preds_baseline)
print(f"   Baseline MSE: {mse_baseline:.5f}")

# ==============================================================================
# 4. PERMUTATION IMPORTANCE LOOP
# ==============================================================================
importances = {}

print("🔄 Running Permutation Importance (this may take a moment)...")

for i, col_name in enumerate(feature_cols):
    # A. Create a copy of the valid test set
    X_test_permuted = X_test_reshaped.copy()
    
    # B. Shuffle the specific feature column (across all samples)
    # Shape is (Samples, 1, Features), so we shuffle index [:, 0, i]
    np.random.shuffle(X_test_permuted[:, 0, i])
    
    # C. Predict with shuffled data
    preds_permuted = model.predict(X_test_permuted, verbose=0)
    
    # D. Calculate new Error
    mse_permuted = mean_squared_error(y_test_scaled, preds_permuted)
    
    # E. Importance = Increase in Error
    # If error went UP, feature is important.
    importance_score = mse_permuted - mse_baseline
    importances[col_name] = importance_score

# ==============================================================================
# 5. NORMALIZE & PLOT
# ==============================================================================
# Convert to DataFrame
importance_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])

# Normalize to Percentage (Sum = 100%)
total_importance = importance_df['Importance'].sum()
importance_df['Importance_Percent'] = (importance_df['Importance'] / total_importance) * 100

# Sort and Take Top 10
importance_df = importance_df.sort_values(by='Importance_Percent', ascending=True).tail(10)

# Colors matching your image (Blue, Orange, Green, Red)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Plotting
plt.figure(figsize=(12, 8))
bars = plt.barh(importance_df['Feature'], importance_df['Importance_Percent'], color=colors)

# Add Labels
plt.xlabel('Importance (%)', fontsize=14)
plt.title('Feature Importance in CNN-BiLSTM Model (Top 10)', fontsize=18, fontweight='bold', pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add Value Labels to ends of bars
for bar in bars:
    width = bar.get_width()
    label_x_pos = width + 0.5
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("✅ Plot saved as 'feature_importance.png'")
plt.show()