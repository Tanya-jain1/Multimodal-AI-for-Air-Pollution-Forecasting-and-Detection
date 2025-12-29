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
SENSOR_ID = 23534
YEARS_TO_DOWNLOAD = range(2016, 2026) # 2016 to 2025

def download_year(year):
    print(f"\n📅 Starting download for YEAR: {year}...")
    
    # Define exact start/end for this year
    params = {
        "datetime_from": f"{year}-01-01T00:00:00Z",
        "datetime_to": f"{year}-12-31T23:59:59Z",
        "limit": 1000,
        "page": 1
    }
    
    url = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements"
    year_rows = []
    page = 1
    max_retries = 3
    
    while True:
        params['page'] = page
        retry_count = 0
        success = False
        
        while retry_count < max_retries:
            try:
                # Increased timeout to 45 seconds to handle the 408 errors
                r = requests.get(url, headers=HEADERS, params=params, timeout=45)
                
                if r.status_code == 200:
                    success = True
                    break
                elif r.status_code == 408:
                    print(f"    ⚠️ Timeout (408) on page {page}. Retrying ({retry_count+1}/{max_retries})...")
                    time.sleep(5) # Wait 5s before retry
                    retry_count += 1
                else:
                    print(f"    ❌ Error {r.status_code} on page {page}.")
                    return None # Stop this year if critical error
            except requests.exceptions.Timeout:
                 print(f"    ⚠️ Connection timed out. Retrying...")
                 retry_count += 1
            except Exception as e:
                print(f"    ❌ Crash: {e}")
                return None

        if not success:
            print(f"    ❌ Failed to get page {page} after {max_retries} attempts. Skipping rest of year.")
            break

        data = r.json().get('results', [])
        if not data:
            break # No more data for this year
            
        for row in data:
            year_rows.append({
                "datetime": row['period']['datetimeFrom']['local'],
                "pm25": row['value']
            })
        
        print(f"    ✅ Year {year}: Collected {len(year_rows)} rows...", end="\r")
        
        if len(data) < 1000:
            break # Last page
            
        page += 1
        time.sleep(0.5) # Gentle pause between pages

    # --- SAVE YEAR TO CSV ---
    if year_rows:
        filename = f"NewDelhi_USPost_{year}.csv"
        df = pd.DataFrame(year_rows)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        df.to_csv(filename)
        print(f"\n    💾 Saved: {filename} ({len(df)} records)")
        return filename
    else:
        print(f"\n    ⚠️ No data found for {year}.")
        return None

# --- MAIN LOOP ---
print(f"🚀 Starting Chunked Download for Sensor {SENSOR_ID}")
saved_files = []

for y in YEARS_TO_DOWNLOAD:
    f = download_year(y)
    if f:
        saved_files.append(f)
    time.sleep(1)

print("\n\n--- MERGING FILES ---")
if saved_files:
    # Combine all year files into one master file
    master_df = pd.concat([pd.read_csv(f) for f in saved_files])
    master_df.sort_values('datetime', inplace=True)
    master_df.to_csv("Ground_Truth_PM25_NewDelhi_USPost_MASTER.csv", index=False)
    print("🎉 ALL DONE! Created 'Ground_Truth_PM25_NewDelhi_USPost_MASTER.csv'")
    
    # Optional: Delete the small yearly files to clean up
    # for f in saved_files: os.remove(f) 
else:
    print("❌ No data collected.")