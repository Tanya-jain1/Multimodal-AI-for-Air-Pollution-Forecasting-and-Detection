import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
openaq_key = os.getenv("OPENAQ_API_KEY")

# Troubleshooting print (OPTIONAL - delete after checking)
if not openaq_key:
    print("Error: OpenAQ Key not found! Check your .env file.")
else:
    print("Keys loaded successfully.")
HEADERS = {"X-API-Key": openaq_key }

# 🗺️ MAPPING: "Location Name" -> [List of IDs found in your CSV]
# I created this based exactly on the list you provided.
STATION_GROUPS = {
    "IGI_Airport": [15, 5650],
    "Anand_Vihar": [235, 5509, 10487],
    "RK_Puram":    [17, 5639, 7044],
    "Punjabi_Bagh":[50, 5540, 6357],
    "Mandir_Marg": [236, 5641, 6358],
    "Pusa":        [5581, 5404, 6356],
    "Jawaharlal_Nehru_Stadium": [6957, 5754],
    "Okhla_Phase_2": [5765, 8239]
}

def download_sensor(sensor_id):
    """Downloads PM2.5 data for a specific Location ID"""
    print(f"      ⬇️ Fetching data for ID {sensor_id}...", end=" ")
    
    # 1. Find the PM2.5 Sensor ID inside this Location
    try:
        url_loc = f"https://api.openaq.org/v3/locations/{sensor_id}/sensors"
        r = requests.get(url_loc, headers=HEADERS, timeout=10)
        sensors = r.json().get('results', [])
        
        pm25_id = None
        for s in sensors:
            if s['parameter']['name'] == 'pm25':
                pm25_id = s['id']
                break
        
        if not pm25_id:
            print("❌ No PM2.5 sensor found.")
            return None

        # 2. Download Data (2020-2025 for speed, extend if needed)
        url_data = f"https://api.openaq.org/v3/sensors/{pm25_id}/measurements"
        params = {
            "datetime_from": "2018-01-01T00:00:00Z", # Adjust start year as needed
            "datetime_to": "2025-12-31T23:59:59Z",
            "limit": 1000,
            "page": 1
        }
        
        all_rows = []
        page = 1
        
        while True:
            params['page'] = page
            r = requests.get(url_data, headers=HEADERS, params=params, timeout=15)
            if r.status_code != 200: break
            
            data = r.json().get('results', [])
            if not data: break
            
            for row in data:
                all_rows.append({
                    "datetime": row['period']['datetimeFrom']['local'],
                    f"pm25_{sensor_id}": row['value'] # Rename col to avoid collision
                })
            
            if len(data) < 1000: break
            page += 1
            # time.sleep(0.1) # removed for speed, add back if 429 errors

        if all_rows:
            df = pd.DataFrame(all_rows)
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df.set_index('datetime', inplace=True)
            df = df[~df.index.duplicated(keep='first')] # Drop duplicates
            print(f"✅ Got {len(df)} rows.")
            return df
        else:
            print("⚠️ No data.")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def process_super_station(name, ids):
    print(f"\n🏭 PROCESSING SUPER-STATION: {name} (IDs: {ids})")
    
    dfs = []
    
    # 1. Download all IDs in the group
    for sid in ids:
        df = download_sensor(sid)
        if df is not None:
            dfs.append(df)
            
    if not dfs:
        print("   ❌ No data retrieved for this group.")
        return

    # 2. Merge them into one wide table
    print("   🔗 Merging timelines...")
    # Join 'outer' ensures we keep data from ID 1 even if ID 2 is missing
    master_df = pd.concat(dfs, axis=1)
    
    # 3. Create the Unified Column (Average of all available sensors)
    # This automatically handles gaps: if sensor A is NaN but B has data, it uses B.
    print("   ⚗️  Calculating Ensemble Average...")
    master_df['pm25_unified'] = master_df.mean(axis=1)
    
    # 4. Clean Up
    final_df = master_df[['pm25_unified']].copy()
    final_df.rename(columns={'pm25_unified': 'pm25'}, inplace=True)
    final_df.sort_index(inplace=True)
    
    # Convert to IST
    final_df.index = final_df.index.tz_convert('Asia/Kolkata')
    
    # 5. Save
    filename = f"Unified_PM25_{name}.csv"
    final_df.to_csv(filename)
    print(f"   🎉 SAVED: {filename} (Records: {len(final_df)})")
    print(f"      Range: {final_df.index.min()} to {final_df.index.max()}")

# --- RUN THE MAIN LOOP ---
if __name__ == "__main__":
    print("--- 🚀 STARTING NETWORK UNIFICATION ---")
    for name, id_list in STATION_GROUPS.items():
        process_super_station(name, id_list)
        time.sleep(2) # Safety pause between groups
    