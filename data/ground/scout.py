"""import requests
import pandas as pd
import time

# --- CONFIGURATION ---
API_KEY = "e227591d7e16193cb7e60bd1e84034efde95d61432a8b2fc470c6072a4edc2cb"
HEADERS = {"X-API-Key": API_KEY}

def find_best_stations():
    print("🛰️ Scanning Delhi for High-Quality Stations...")
    
    # 1. Search for locations in Delhi
    url = "https://api.openaq.org/v3/locations"
    params = {
        "limit": 100,
        "page": 1,
        "coordinates": "28.61,77.20", # Center of Delhi
        "radius": 25000, # 25km radius
        "order_by": "id"
    }
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if response.status_code != 200:
            print(f"❌ Initial Search Failed: {response.status_code}")
            return
            
        locations = response.json().get('results', [])
        print(f"   Found {len(locations)} potential stations. Checking sensor counts...\n")
        
    except Exception as e:
        print(f"❌ Network Error during initial search: {e}")
        return

    station_stats = []
    
    # 2. Check each station specifically
    for i, loc in enumerate(locations):
        loc_id = loc['id']
        name = loc['name']
        
        # Simple progress indicator
        print(f"   [{i+1}/{len(locations)}] Checking {name[:20]}...", end="", flush=True)
        
        sens_url = f"https://api.openaq.org/v3/locations/{loc_id}/sensors"
        
        try:
            # --- THE FIX: Pause to respect API limits ---
            time.sleep(0.3) 
            
            sens_resp = requests.get(sens_url, headers=HEADERS, timeout=10)
            
            if sens_resp.status_code == 200:
                sensors = sens_resp.json().get('results', [])
                
                # Check for critical pollutants
                has_pm25 = any(s['parameter']['name'] == 'pm25' for s in sensors)
                has_pm10 = any(s['parameter']['name'] == 'pm10' for s in sensors)
                has_no2  = any(s['parameter']['name'] == 'no2'  for s in sensors)
                has_o3   = any(s['parameter']['name'] == 'o3'   for s in sensors)
                has_co   = any(s['parameter']['name'] == 'co'   for s in sensors)
                
                # Count total sensors found
                count = len(sensors)
                
                # We only want stations that have AT LEAST PM2.5
                if has_pm25:
                    print(" ✅")
                    station_stats.append({
                        "ID": loc_id,
                        "Name": name,
                        "Sensors_Count": count,
                        "PM2.5": "Yes" if has_pm25 else "No",
                        "PM10": "Yes" if has_pm10 else "No",
                        "NO2": "Yes" if has_no2 else "No",
                        "Ozone": "Yes" if has_o3 else "No",
                        "CO": "Yes" if has_co else "No"
                    })
                else:
                    print(" ⚠️ No PM2.5")
            
            elif sens_resp.status_code == 429:
                print(" ⏳ Rate Limit! Sleeping 5s...")
                time.sleep(5)
            else:
                print(f" ⚠️ Status {sens_resp.status_code}")

        except Exception as e:
            print(f" ❌ Error: {e}")
            continue

    # 3. Present Results
    if not station_stats:
        print("\n❌ No suitable stations found.")
        return

    df = pd.DataFrame(station_stats)
    
    # Sort by "Richness" (Stations with most sensor variety first)
    # We create a temporary score: +1 for each pollutant present
    df['Score'] = (df['PM2.5']=='Yes').astype(int) + \
                  (df['PM10']=='Yes').astype(int) + \
                  (df['NO2']=='Yes').astype(int) + \
                  (df['Ozone']=='Yes').astype(int) + \
                  (df['CO']=='Yes').astype(int)
                  
    df = df.sort_values(by='Score', ascending=False)
    
    print("\n🏆 BEST CANDIDATES FOR YOUR MODEL (Sorted by Quality):")
    # Display clean table
    print(df[['ID', 'Name', 'PM2.5', 'PM10', 'NO2', 'Ozone', 'CO']].to_string(index=False))
    
    # Save for reference
    df.to_csv("Delhi_Station_Candidates.csv", index=False)
    print("\n💾 Saved list to 'Delhi_Station_Candidates.csv'")

find_best_stations()"""

"""import requests
import pandas as pd
import time

API_KEY = "e227591d7e16193cb7e60bd1e84034efde95d61432a8b2fc470c6072a4edc2cb"
HEADERS = {"X-API-Key": API_KEY}

def find_survivor_stations():
    print("🕵️ Scanning for 'Survivor' Stations (Continuous Data 2020-2025)...")
    
    # 1. Get all Delhi locations
    url = "https://api.openaq.org/v3/locations"
    params = {
        "limit": 100,
        "coordinates": "28.61,77.20",
        "radius": 25000,
        "order_by": "id"
    }
    
    locations = requests.get(url, headers=HEADERS, params=params).json().get('results', [])
    print(f"   Found {len(locations)} stations to check.\n")
    
    survivors = []

    # 2. Check every single station
    for i, loc in enumerate(locations):
        loc_id = loc['id']
        name = loc['name']
        
        print(f"   [{i+1}/{len(locations)}] Auditing {name[:25]}... ", end="", flush=True)
        
        try:
            # Get sensors
            s_url = f"https://api.openaq.org/v3/locations/{loc_id}/sensors"
            sensors = requests.get(s_url, headers=HEADERS, timeout=10).json().get('results', [])
            
            # Check for Continuity
            has_history = False
            has_recent = False
            best_pm25_count = 0
            
            for s in sensors:
                if s['parameter']['name'] == 'pm25':
                    first = s.get('datetimeFirst', {}).get('local', '9999')
                    last = s.get('datetimeLast', {}).get('local', '0000')
                    count = 0
                    if 'coverage' in s and s['coverage']:
                        count = s['coverage'].get('observedCount', 0)

                    # Criteria: Starts before 2021 AND Ends in 2025
                    if first[:4] < '2021' and last[:4] == '2025':
                        has_history = True
                        has_recent = True
                        best_pm25_count = count
            
            if has_history and has_recent:
                print(f"✅ SURVIVOR! ({best_pm25_count} pts)")
                survivors.append({
                    "Name": name,
                    "ID": loc_id,
                    "PM2.5_Count": best_pm25_count
                })
            else:
                print("❌ Fragmented")
                
            time.sleep(0.2)
            
        except Exception:
            print("⚠️ Error")

    # 3. Report
    print("\n🏆 THE CHOSEN ONES (Use these for your paper):")
    if survivors:
        df = pd.DataFrame(survivors)
        print(df.to_string(index=False))
    else:
        print("No single station has a perfect continuous line. We may need to merge 'Old' + 'New'.")

find_survivor_stations()"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# 1. LOAD YOUR DATA (Assuming a big DataFrame with 'timestamp', 'station_id', 'pm25')
# df = pd.read_csv("your_data.csv") 

# 2. CHECK THE REAL DAMAGE
# Pivot to see stations side-by-side
df_pivot = df.pivot_table(index='timestamp', columns='station_name', values='PM2.5')

# Calculate percentage of missing data per station
missing_stats = df_pivot.isnull().mean() * 100
usable_stations = missing_stats[missing_stats < 50].sort_values() # Keep stations with <50% missing

print("--- USABLE STATIONS (Less than 50% missing) ---")
print(usable_stations)

# 3. SELECT YOUR "CHOSEN ONES"
# Pick the best 5-8 stations from the list above + Your Target (New Delhi)
selected_stations = usable_stations.index.tolist() 
# Ensure 'New Delhi' is in the list
if 'New Delhi' not in selected_stations: 
    selected_stations.append('New Delhi')

df_selected = df_pivot[selected_stations]

# 4. IMPUTE (FILL GAPS)
# KNN Imputer is best for spatial sensor data
imputer = KNNImputer(n_neighbors=5) 
df_clean_array = imputer.fit_transform(df_selected)
df_clean = pd.DataFrame(df_clean_array, columns=selected_stations, index=df_selected.index)

print("--- DATA CLEANED ---")
print(f"Original Shape: {df_pivot.shape}")
print(f"Cleaned Shape: {df_clean.shape}")

# Now you can Normalise this 'df_clean' and feed it to your CNN-LSTM