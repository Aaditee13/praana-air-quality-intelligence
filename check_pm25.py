from dotenv import load_dotenv
load_dotenv()

from src import data_sources as ds

readings, err = ds.fetch_latest_readings(28.6139, 77.2090, radius_m=25000)
print("error:", err)
print("pm25 raw value:", readings.get("pm25") if readings else None)