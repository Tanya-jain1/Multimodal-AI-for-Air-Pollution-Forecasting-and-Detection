from flask import Flask, render_template
import requests
import pandas as pd
import math
import joblib
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# 🟢  YOUR WAQI TOKEN HERE
WAQI_TOKEN = "3851d5b7d5bb01c711b9ea24ea0a03f6154d0f64" 

# 🔵 YOUR OPENWEATHERMAP KEY HERE
OWM_KEY = "1b636ed3f2c75bb2ce34b0fe461bf910" 

CITY_NAME = "Delhi"

print("⏳ Starting Hybrid Web Server...")
try:
    model = load_model('best_cnn_bilstm_model.keras')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    with open('model_features.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    print("✅ System Ready!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# ==============================================================================
# 2. HELPER: AQI CALCULATORS
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
    if aqi <= 50: return "Good", "#2ecc71"
    elif aqi <= 100: return "Satisfactory", "#a3e059"
    elif aqi <= 200: return "Moderate", "#f1c40f"
    elif aqi <= 300: return "Poor", "#e67e22"
    elif aqi <= 400: return "Very Poor", "#e74c3c"
    else: return "Severe", "#8e44ad"

# ==============================================================================
# 3. HELPER: HYBRID DATA FETCHER
# ==============================================================================
def get_hybrid_prediction():
    try:
        data = {}
        
        # --- A. FETCH GROUND TRUTH (WAQI) ---
        waqi_url = f"https://api.waqi.info/feed/{CITY_NAME}/?token={WAQI_TOKEN}"
        waqi_res = requests.get(waqi_url).json()
        
        if waqi_res['status'] != 'ok': return None, None
        
        station_data = waqi_res['data']['iaqi']
        
        # Pollution
        data['PM2.5'] = station_data.get('pm25', {}).get('v', 0)
        data['PM10']  = station_data.get('pm10', {}).get('v', 0)
        data['NO2']   = station_data.get('no2', {}).get('v', 0)
        data['CO']    = station_data.get('co', {}).get('v', 0)
        data['SO2']   = station_data.get('so2', {}).get('v', 0)
        data['O3']    = station_data.get('o3', {}).get('v', 0)
        
        # Weather (Prefer Ground Station)
        data['Temperature'] = station_data.get('t', {}).get('v', 0)
        data['Humidity']    = station_data.get('h', {}).get('v', 0)
        data['WindSpeed']   = station_data.get('w', {}).get('v', 0)
        
        # --- B. FETCH MISSING DATA (OWM) ---
        lat = waqi_res['data']['city']['geo'][0]
        lon = waqi_res['data']['city']['geo'][1]
        
        weather_res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OWM_KEY}").json()
        poll_res = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OWM_KEY}").json()

        # Fill Gaps
        if 'p' not in station_data: data['WindSpeed'] = weather_res['wind']['speed']
        
        wind_deg = weather_res['wind'].get('deg', 0)
        data['Wind_X'] = data['WindSpeed'] * math.cos(math.radians(wind_deg))
        data['Wind_Y'] = data['WindSpeed'] * math.sin(math.radians(wind_deg))
        
        data['Rain'] = weather_res.get('rain', {}).get('1h', 0.0)
        data['NH3'] = poll_res['list'][0]['components']['nh3']
        data['NO'] = poll_res['list'][0]['components']['no']

        # Calculated Features
        data['AQI_Lag1'] = data['PM2.5']
        data['AQI_Lag24'] = data['PM2.5']
        data['PBL_Height'] = 800.0
        data['Ventilation'] = data['PBL_Height'] * data['WindSpeed']
        data['Total_FRP'] = 0.0
        
        # --- C. PREDICTION ---
        model_input = {k: data.get(k, 0) for k in feature_cols}
        # Case Insensitive Match
        for k in feature_cols:
            if k not in model_input:
                found = next((val for key, val in data.items() if key.lower() == k.lower()), 0)
                model_input[k] = found

        input_df = pd.DataFrame([model_input])[feature_cols]
        
        input_scaled = scaler_X.transform(input_df)
        input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
        
        pred_scaled = model.predict(input_reshaped, verbose=0)
        pred_pm25 = scaler_y.inverse_transform(pred_scaled)[0][0]
        if pred_pm25 < 0: pred_pm25 = 0
        
        # --- D. FINAL AQI ---
        curr_aqi = max(calc_pm25_aqi(data['PM2.5']), calc_pm10_aqi(data['PM10']))
        pred_aqi = max(calc_pm25_aqi(pred_pm25), calc_pm10_aqi(data['PM10']))
        
        return curr_aqi, pred_aqi

    except Exception as e:
        print(f"Error: {e}")
        return None, None

# ==============================================================================
# 4. ROUTE
# ==============================================================================
@app.route('/')
def dashboard():
    current, forecast = get_hybrid_prediction()
    
    if current is None:
        return "<h1>Error fetching data. Check Internet/Tokens.</h1>"
    
    status, color = get_aqi_category(forecast)
    
    return render_template('index.html', 
                           city=CITY_NAME,
                           current=int(current),
                           forecast=int(forecast),
                           status=status,
                           color=color)

if __name__ == '__main__':
    app.run(debug=True)