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

def repair_2024():
    print("🚑 Attempting to REPAIR 2024 data...")
    
    # We split 2024 into two halves to reduce server load
    periods = [
        ("2024-01-01T00:00:00Z", "2024-06-30T23:59:59Z"),
        ("2024-07-01T00:00:00Z", "2024-12-31T23:59:59Z")
    ]
    
    year_rows = []
    
    for start, end in periods:
        print(f"   Downloading chunk: {start[:10]} to {end[:10]}")
        params = {
            "datetime_from": start,
            "datetime_to": end,
            "limit": 1000,
            "page": 1
        }
        
        url = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements"
        page = 1
        
        while True:
            params['page'] = page
            try:
                r = requests.get(url, headers=HEADERS, params=params, timeout=30)
                
                if r.status_code != 200:
                    print(f"      ⚠️ Error {r.status_code} on page {page}. Retrying...")
                    time.sleep(5)
                    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
                    if r.status_code != 200:
                         print("      ❌ Skipped a batch due to server error.")
                         break
                
                data = r.json().get('results', [])
                if not data: break
                
                for row in data:
                    year_rows.append({
                        "datetime": row['period']['datetimeFrom']['local'],
                        "pm25": row['value']
                    })
                
                if len(data) < 1000: break
                page += 1
                time.sleep(0.5)
                
            except Exception as e:
                print(f"      ❌ Crash: {e}")
                break
                
    # Save the file
    if year_rows:
        df = pd.DataFrame(year_rows)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        filename = "NewDelhi_USPost_2024.csv"
        df.to_csv(filename)
        print(f"✅ SUCCESS: Recovered {len(df)} records for 2024.")
        return True
    else:
        print("❌ Failed to recover 2024.")
        return False

# --- RE-MERGE EVERYTHING ---
def remerge_all():
    print("\n🔄 Re-merging all CSV files...")
    
    # List all years including the new 2024 file
    files = [f"NewDelhi_USPost_{year}.csv" for year in range(2016, 2026)]
    
    # Filter out files that don't exist (just in case)
    valid_files = [f for f in files if os.path.exists(f)]
    
    if not valid_files:
        print("No files found to merge!")
        return

    # Combine
    master_df = pd.concat([pd.read_csv(f) for f in valid_files])
    
    # Clean and Sort
    master_df['datetime'] = pd.to_datetime(master_df['datetime']) # Keep as datetime
    master_df.sort_values('datetime', inplace=True)
    
    output = "Ground_Truth_PM25_NewDelhi_USPost_MASTER.csv"
    master_df.to_csv(output, index=False)
    
    print(f"🎉 FINAL COMPLETE DATASET: {output}")
    print(f"📅 Total Records: {len(master_df)}")
    print(f"📅 Time Range: {master_df['datetime'].min()} to {master_df['datetime'].max()}")

if __name__ == "__main__":
    if repair_2024():
        remerge_all()