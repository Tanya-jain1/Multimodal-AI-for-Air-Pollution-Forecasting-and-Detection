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
HEADERS = {"X-API-Key": openaq_key}

# 🎯 NEW TARGETS ONLY
# We only list the ones you haven't downloaded yet.
STATION_GROUPS = {
    "Bawana_Industrial":        [8472],             
    "Karni_Singh_GreenBelt":    [6934, 5744]
}

def download_sensor(sensor_id):
    """Downloads PM2.5 data for a specific Location ID"""
    print(f"      ⬇️ Fetching data for ID {sensor_id}...", end=" ")
    
    try:
        # 1. Find PM2.5 Sensor
        url_loc = f"https://api.openaq.org/v3/locations/{sensor_id}/sensors"
        r = requests.get(url_loc, headers=HEADERS, timeout=15)
        sensors = r.json().get('results', [])
        
        pm25_id = None
        for s in sensors:
            if s['parameter']['name'] == 'pm25':
                pm25_id = s['id']
                break
        
        if not pm25_id:
            print("❌ No PM2.5 sensor found.")
            return None

        # 2. Download Data
        # Bawana and Karni Singh might be newer, but we request from 2018 to be safe
        url_data = f"https://api.openaq.org/v3/sensors/{pm25_id}/measurements"
        params = {
            "datetime_from": "2018-01-01T00:00:00Z",
            "datetime_to": "2025-12-31T23:59:59Z",
            "limit": 1000,
            "page": 1
        }
        
        all_rows = []
        page = 1
        
        while True:
            params['page'] = page
            r = requests.get(url_data, headers=HEADERS, params=params, timeout=30)
            if r.status_code != 200: 
                # Simple retry logic for 408/500 errors
                time.sleep(2)
                r = requests.get(url_data, headers=HEADERS, params=params, timeout=30)
                if r.status_code != 200: break

            data = r.json().get('results', [])
            if not data: break
            
            for row in data:
                all_rows.append({
                    "datetime": row['period']['datetimeFrom']['local'],
                    f"pm25_{sensor_id}": row['value']
                })
            
            # Print progress dots every 5 pages
            if page % 5 == 0: print(".", end="", flush=True)

            if len(data) < 1000: break
            page += 1

        if all_rows:
            df = pd.DataFrame(all_rows)
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df.set_index('datetime', inplace=True)
            df = df[~df.index.duplicated(keep='first')] # Drop duplicates
            print(f" ✅ Got {len(df)} rows.")
            return df
        else:
            print(" ⚠️ No data.")
            return None

    except Exception as e:
        print(f" ❌ Error: {e}")
        return None

def process_super_station(name, ids):
    print(f"\n🏭 PROCESSING SUPER-STATION: {name} (IDs: {ids})")
    
    dfs = []
    for sid in ids:
        df = download_sensor(sid)
        if df is not None:
            dfs.append(df)
            
    if not dfs:
        print("   ❌ No data retrieved for this group.")
        return

    print("   🔗 Merging timelines...")
    master_df = pd.concat(dfs, axis=1)
    
    # Ensemble Average (handles the 2 Karni Singh sensors automatically)
    master_df['pm25_unified'] = master_df.mean(axis=1)
    
    final_df = master_df[['pm25_unified']].copy()
    final_df.rename(columns={'pm25_unified': 'pm25'}, inplace=True)
    final_df.sort_index(inplace=True)
    final_df.index = final_df.index.tz_convert('Asia/Kolkata')
    
    filename = f"Unified_PM25_{name}.csv"
    final_df.to_csv(filename)
    print(f"   🎉 SAVED: {filename} (Records: {len(final_df)})")
    print(f"      Range: {final_df.index.min()} to {final_df.index.max()}")

if __name__ == "__main__":
    for name, id_list in STATION_GROUPS.items():
        process_super_station(name, id_list)
        time.sleep(2)