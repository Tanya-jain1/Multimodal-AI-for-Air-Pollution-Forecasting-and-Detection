import requests
import pandas as pd
import numpy as np
import joblib
import pickle
import math 
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Load the keys securely
# 🔵 OWM KEY - KEEP THIS FOR WEATHER BACKUP
owm_key = os.getenv("OWM_KEY")
# 🟢 WAQI (AQICN) TOKEN - GET THIS FROM EMAIL
waqi_token = os.getenv("WAQI_TOKEN")

# Troubleshooting print (OPTIONAL - delete after checking)
if not openaq_key:
    print("Error: OpenAQ Key not found! Check your .env file.")
else:
    print("Keys loaded successfully.")

# CITY: For WAQI, you can use "delhi" (average) or specific station like "anand-vihar"
CITY_NAME = "Delhi"       

MODEL_PATH = 'best_cnn_bilstm_model.keras'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
FEATURES_PATH = 'model_features.pkl'

# ==============================================================================
# 2. HELPER: HYBRID DATA FETCHER
# ==============================================================================
def get_hybrid_data():
    print(f"🌍 Connecting to CPCB Stations via WAQI...")
    
    data = {}
    
    # --- A. FETCH GROUND TRUTH (POLLUTION) FROM WAQI ---
    try:
        # Use 'here' to locate based on IP, or 'delhi' for city average
        waqi_url = f"https://api.waqi.info/feed/{CITY_NAME}/?token={WAQI_TOKEN}"
        waqi_res = requests.get(waqi_url).json()
        
        if waqi_res['status'] != 'ok':
            print("❌ WAQI Error. Check Token.")
            return None

        station_data = waqi_res['data']['iaqi']
        
        # Extract available sensors (Some might be missing!)
        # WAQI values are already AQI-like or raw conc. usually raw for gases.
        data['PM2.5'] = station_data.get('pm25', {}).get('v', 0)
        data['PM10']  = station_data.get('pm10', {}).get('v', 0)
        data['NO2']   = station_data.get('no2', {}).get('v', 0)
        data['CO']    = station_data.get('co', {}).get('v', 0)
        data['SO2']   = station_data.get('so2', {}).get('v', 0)
        data['O3']    = station_data.get('o3', {}).get('v', 0)
        
        # Weather from Ground Station (Preferred)
        data['Temperature'] = station_data.get('t', {}).get('v', 0)
        data['Humidity']    = station_data.get('h', {}).get('v', 0)
        data['WindSpeed']   = station_data.get('w', {}).get('v', 0)
        
        print(f"✅ Ground Data Received! Current PM2.5: {data['PM2.5']}")

    except Exception as e:
        print(f"❌ Error fetching WAQI: {e}")
        return None

    # --- B. FETCH MISSING DATA FROM OWM (BACKUP) ---
    # WAQI often misses: NH3, NO, Rain, specific Wind Direction
    try:
        # Get coords from WAQI result to sync locations
        lat = waqi_res['data']['city']['geo'][0]
        lon = waqi_res['data']['city']['geo'][1]
        
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OWM_KEY}"
        poll_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OWM_KEY}"
        
        weather_res = requests.get(weather_url).json()
        poll_res = requests.get(poll_url).json()
        
        # Fill Gaps
        if 'p' not in station_data: 
             data['WindSpeed'] = weather_res['wind']['speed'] # Fallback if WAQI wind missing
        
        # Wind Direction Math
        wind_deg = weather_res['wind'].get('deg', 0)
        wind_rad = math.radians(wind_deg)
        data['Wind_X'] = data['WindSpeed'] * math.cos(wind_rad)
        data['Wind_Y'] = data['WindSpeed'] * math.sin(wind_rad)
        
        # Rain
        if 'rain' in weather_res and '1h' in weather_res['rain']:
            data['Rain'] = weather_res['rain']['1h']
        else:
            data['Rain'] = 0.0
            
        # Gases OWM has but WAQI might miss
        owm_comps = poll_res['list'][0]['components']
        data['NH3'] = owm_comps['nh3'] # Important for agricultural pollution
        data['NO']  = owm_comps['no']  # Often missing in CPCB public feeds

    except Exception as e:
        print(f"⚠️ OWM Backup Warning: {e}")
        # Defaults if OWM fails
        data['Wind_X'] = 0; data['Wind_Y'] = 0; data['Rain'] = 0; data['NH3'] = 0; data['NO'] = 0

    # --- C. CALCULATED FEATURES ---
    data['AQI_Lag1'] = data['PM2.5']
    data['AQI_Lag24'] = data['PM2.5']
    data['PBL_Height'] = 800.0
    data['Ventilation'] = data['PBL_Height'] * data['WindSpeed']
    data['Total_FRP'] = 0.0

    return data

# ==============================================================================
# 3. HELPER: AQI CALC & DISPLAY
# ==============================================================================
def calc_pm25_aqi(pm25):
    if pm25 <= 30: return pm25 * (50/30)
    elif pm25 <= 60: return 50 + (pm25 - 30) * (50/30)
    elif pm25 <= 90: return 100 + (pm25 - 60) * (100/30)
    elif pm25 <= 120: return 200 + (pm25 - 90) * (100/30)
    elif pm25 <= 250: return 300 + (pm25 - 120) * (100/130)
    else: return 400 + (pm25 - 250) * (100/130)

def calc_pm10_aqi(pm10):
    if pm10 <= 50: return pm10
    elif pm10 <= 100: return pm10
    elif pm10 <= 250: return 100 + (pm10 - 100) * (100/150)
    elif pm10 <= 350: return 200 + (pm10 - 250) * (100/100)
    elif pm10 <= 430: return 300 + (pm10 - 350) * (100/80)
    else: return 400 + (pm10 - 430) * (100/70)

def get_aqi_category(aqi):
    if aqi <= 50: return "Good 🟢"
    elif aqi <= 100: return "Satisfactory 🟢"
    elif aqi <= 200: return "Moderate 🟡"
    elif aqi <= 300: return "Poor 🟠"
    elif aqi <= 400: return "Very Poor 🔴"
    else: return "Severe 🟣"

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
print("⏳ Loading System...")
try:
    model = load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    with open(FEATURES_PATH, 'rb') as f:
        feature_cols = pickle.load(f)
except Exception: exit()

live_data = get_hybrid_data()

if live_data:
    # Prepare Input
    model_input = {k: live_data.get(k, 0) for k in feature_cols}
    # Handle casing mismatches
    for k in feature_cols:
        if k not in model_input:
             # Try finding key in live_data regardless of case
             found = next((val for key, val in live_data.items() if key.lower() == k.lower()), 0)
             model_input[k] = found

    input_df = pd.DataFrame([model_input])[feature_cols]

    # Predict
    input_scaled = scaler_X.transform(input_df)
    input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    prediction_scaled = model.predict(input_reshaped, verbose=0)
    predicted_pm25 = scaler_y.inverse_transform(prediction_scaled)[0][0]
    if predicted_pm25 < 0: predicted_pm25 = 0

    # Display Results
    curr_pm25 = live_data['PM2.5']
    curr_pm10 = live_data['PM10']
    
    curr_aqi = max(calc_pm25_aqi(curr_pm25), calc_pm10_aqi(curr_pm10))
    pred_aqi = max(calc_pm25_aqi(predicted_pm25), calc_pm10_aqi(curr_pm10))

    print("\n" + "="*50)
    print(f"🌆 LIVE GROUND STATION DATA: {CITY_NAME.upper()}")
    print("="*50)
    print(f"📉 Station PM2.5: {curr_pm25} (Real CPCB Data)")
    print(f"📉 Station PM10:  {curr_pm10} (Real CPCB Data)")
    print("-" * 50)
    print(f"🔮 Model Prediction (Next Hour PM2.5): {predicted_pm25:.2f}")
    print("-" * 50)
    print(f"🚦 Current AQI:   {int(curr_aqi)} ({get_aqi_category(curr_aqi)})")
    print(f"🚀 Predicted AQI: {int(pred_aqi)} ({get_aqi_category(pred_aqi)})")
    print("="*50)