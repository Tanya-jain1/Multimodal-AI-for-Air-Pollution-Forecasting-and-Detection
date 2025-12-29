import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # pip install xgboost

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
FILE_NAME = 'Final_Model_Data.csv'
TEST_SPLIT = 0.2

print("⏳ Loading Data for Classic ML Models...")
df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)

feature_cols = [c for c in df.columns if c != 'pm25']
target_col = 'pm25'
X = df[feature_cols].values
y = df[target_col].values

print(f"📊 Features used: {feature_cols}")

# ==============================================================================
# 2. SPLITTING (Strictly by Time)
# ==============================================================================
# Note: Tree models (RF/XGB) do NOT require 0-1 scaling. We use raw data.
split_idx = int(len(df) * (1 - TEST_SPLIT))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"🚂 Training Rows: {len(X_train)}")
print(f"🧪 Testing Rows:  {len(X_test)}")

# ==============================================================================
# 3. TRAIN RANDOM FOREST 🌳
# ==============================================================================
print("\n🧠 Training Random Forest (Model A)...")
# n_estimators=100: Build 100 trees
# n_jobs=-1: Use all CPU cores (Faster)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluate RF
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = math.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# ==============================================================================
# 4. TRAIN XGBOOST 🚀
# ==============================================================================
print("\n🧠 Training XGBoost (Model B)...")
# XGBoost is usually faster and more accurate than Random Forest
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, early_stopping_rounds=10)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Predict
xgb_pred = xgb_model.predict(X_test)

# Evaluate XGB
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_rmse = math.sqrt(xgb_mse)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# ==============================================================================
# 5. FINAL COMPARISON TABLE
# ==============================================================================
print("\n" + "="*60)
print(f"{'METRIC':<15} | {'RANDOM FOREST':<18} | {'XGBOOST':<18}")
print("="*60)
print(f"{'RMSE':<15} | {rf_rmse:<18.2f} | {xgb_rmse:<18.2f}")
print(f"{'MAE':<15} | {rf_mae:<18.2f} | {xgb_mae:<18.2f}")
print(f"{'R²':<15} | {rf_r2:<18.4f} | {xgb_r2:<18.4f}")
print("="*60)

# ==============================================================================
# 6. VISUALIZATION: FEATURE IMPORTANCE (Thesis Gold) 🌟
# ==============================================================================
# This plot proves scientifically WHICH factors drive pollution
plt.figure(figsize=(12, 6))

# Use XGBoost importance (usually more reliable for time-series)
importance = xgb_model.feature_importances_
sorted_idx = np.argsort(importance)

plt.barh(np.array(feature_cols)[sorted_idx], importance[sorted_idx], color='teal')
plt.title("What Drives Delhi's Pollution? (XGBoost Feature Importance)")
plt.xlabel("Relative Importance Score")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==============================================================================
# 7. SAVE MODELS
# ==============================================================================
print("💾 Saving Models...")
joblib.dump(rf_model, 'best_random_forest.pkl')
xgb_model.save_model('best_xgboost.json')
print("✅ Saved 'best_random_forest.pkl' and 'best_xgboost.json'")