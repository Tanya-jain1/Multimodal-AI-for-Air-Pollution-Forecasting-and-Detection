import joblib
import pickle
import pandas as pd
import numpy as np
import math
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, classification_report
)

# ==============================================================================
# CONFIGURATION — match these exactly to your training script
# ==============================================================================
FILE_NAME        = 'Final_Model_Data.csv'
MODEL_PATH       = 'best_cnn_bilstm_model.keras'
SCALER_X_PATH    = 'scaler_X.pkl'
SCALER_Y_PATH    = 'scaler_y.pkl'
FEATURES_PATH    = 'model_features.pkl'
TEST_SPLIT       = 0.2

# --- PM2.5 thresholds for classification metrics ---
# Adjust these breakpoints to match your use-case / AQI standard
THRESHOLDS = {
    "Good":       (0,   12),
    "Moderate":   (12,  35.4),
    "Unhealthy":  (35.4, 55.4),
    "Very Unhealthy": (55.4, 150.4),
    "Hazardous":  (150.4, float('inf'))
}

def pm25_to_category(values: np.ndarray) -> np.ndarray:
    """Map continuous PM2.5 values → integer category labels."""
    bins   = [0, 12, 35.4, 55.4, 150.4, float('inf')]
    labels = [0, 1,  2,    3,    4]
    cats   = np.zeros(len(values), dtype=int)
    for i, v in enumerate(values.flatten()):
        for j in range(len(bins) - 1):
            if bins[j] <= v < bins[j + 1]:
                cats[i] = labels[j]
                break
    return cats

# ==============================================================================
# 1. LOAD SAVED ARTIFACTS
# ==============================================================================
print("📦 Loading saved model and scalers...")
model     = load_model(MODEL_PATH)
scaler_X  = joblib.load(SCALER_X_PATH)
scaler_y  = joblib.load(SCALER_Y_PATH)

with open(FEATURES_PATH, 'rb') as f:
    feature_cols = pickle.load(f)

print(f"✅ Model loaded  : {MODEL_PATH}")
print(f"✅ Features used : {feature_cols}")

# ==============================================================================
# 2. LOAD & SPLIT DATA  (identical logic to training script)
# ==============================================================================
print("\n⏳ Loading dataset...")
df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)

X = df[feature_cols].values
y = df['pm25'].values.reshape(-1, 1)

split_idx   = int(len(df) * (1 - TEST_SPLIT))
X_train_raw = X[:split_idx];  y_train_raw = y[:split_idx]
X_test_raw  = X[split_idx:];  y_test_raw  = y[split_idx:]

print(f"🚂 Train rows : {len(X_train_raw)}")
print(f"🧪 Test  rows : {len(X_test_raw)}")

# ==============================================================================
# 3. SCALE & RESHAPE
# ==============================================================================
X_train_s = scaler_X.transform(X_train_raw)
X_test_s  = scaler_X.transform(X_test_raw)
y_train_s = scaler_y.transform(y_train_raw)
y_test_s  = scaler_y.transform(y_test_raw)

# Shape → (Samples, TimeSteps=1, Features)
X_train_r = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
X_test_r  = X_test_s.reshape((X_test_s.shape[0],  1, X_test_s.shape[1]))

# ==============================================================================
# 4. PREDICT
# ==============================================================================
print("\n🔮 Running predictions (no re-training)...")
train_pred_s = model.predict(X_train_r, verbose=0)
test_pred_s  = model.predict(X_test_r,  verbose=0)

# Inverse-transform back to original PM2.5 scale
y_train_act  = scaler_y.inverse_transform(y_train_s)
y_test_act   = scaler_y.inverse_transform(y_test_s)
train_pred   = scaler_y.inverse_transform(train_pred_s)
test_pred    = scaler_y.inverse_transform(test_pred_s)

# ==============================================================================
# 5. REGRESSION METRICS  (R², RMSE, MAE)
# ==============================================================================
def regression_metrics(y_true, y_pred, label=""):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n📈 REGRESSION METRICS — {label}")
    print("-" * 45)
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print("-" * 45)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

reg_train = regression_metrics(y_train_act, train_pred, "TRAIN SET")
reg_test  = regression_metrics(y_test_act,  test_pred,  "TEST SET")

# ==============================================================================
# 6. CLASSIFICATION METRICS  (Precision, Recall, F1)
#    PM2.5 values are bucketed into AQI categories first
# ==============================================================================
def classification_metrics(y_true, y_pred, label=""):
    y_true_cat = pm25_to_category(y_true)
    y_pred_cat = pm25_to_category(y_pred)

    cat_names = list(THRESHOLDS.keys())

    precision = precision_score(y_true_cat, y_pred_cat, average='weighted',
                                zero_division=0)
    recall    = recall_score(y_true_cat, y_pred_cat, average='weighted',
                             zero_division=0)
    f1        = f1_score(y_true_cat, y_pred_cat, average='weighted',
                         zero_division=0)

    print(f"\n🎯 CLASSIFICATION METRICS — {label}")
    print("-" * 45)
    print(f"  Weighted Precision : {precision:.4f}")
    print(f"  Weighted Recall    : {recall:.4f}")
    print(f"  Weighted F1-Score  : {f1:.4f}")
    print("-" * 45)
    print("\n  Per-class breakdown:")
    # Only show classes that actually appear in y_true
    present = sorted(set(y_true_cat))
    print(classification_report(
        y_true_cat, y_pred_cat,
        labels=present,
        target_names=[cat_names[i] for i in present],
        zero_division=0
    ))
    return {"Precision": precision, "Recall": recall, "F1": f1}

cls_train = classification_metrics(y_train_act, train_pred, "TRAIN SET")
cls_test  = classification_metrics(y_test_act,  test_pred,  "TEST SET")

# ==============================================================================
# 7. COMBINED SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 65)
print(f"{'METRIC':<22} | {'TRAIN':>18} | {'TEST':>18}")
print("=" * 65)
for metric in ["RMSE", "MAE", "R2"]:
    print(f"{metric:<22} | {reg_train[metric]:>18.4f} | {reg_test[metric]:>18.4f}")
print("-" * 65)
for metric in ["Precision", "Recall", "F1"]:
    print(f"{metric:<22} | {cls_train[metric]:>18.4f} | {cls_test[metric]:>18.4f}")
print("=" * 65)
print("\n✅ Evaluation complete — no retraining was performed.")