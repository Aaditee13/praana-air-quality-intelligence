"""
demo_data.py
------------
Offline fallback data so the dashboard is always demoable — even with no
internet, no OpenAQ key, or a venue wifi that drops mid-pitch. "Demo data"
mode in the sidebar routes here instead of to data_sources.py's live calls.
Every number here is illustrative, not a live reading, and the UI says so.
"""

from datetime import datetime, timedelta
from typing import Dict, List

DEMO_READINGS: Dict[str, float] = {
    "pm25": 180, "pm10": 220, "no2": 85, "so2": 12, "co": 2.1,
    "_station_name": "Demo Station (offline sample data)",
}

DEMO_OSM_SITES: List[Dict] = [
    {"name": "Construction site, Sector 15", "lat": 28.713, "lon": 77.101, "category": "Construction"},
    {"name": "NH48 corridor", "lat": 28.551, "lon": 77.071, "category": "Vehicular"},
    {"name": "Industrial cluster, Bawana", "lat": 28.799, "lon": 77.034, "category": "Industrial"},
]

DEMO_MULTICITY: List[Dict] = [
    {"city": "Delhi NCR", "aqi": 312, "category": "Very Poor", "dominant_pollutant": "pm25",
     "station": "Demo Station", "error": None},
    {"city": "Mumbai", "aqi": 142, "category": "Moderate", "dominant_pollutant": "pm10",
     "station": "Demo Station", "error": None},
    {"city": "Kolkata", "aqi": 168, "category": "Moderate", "dominant_pollutant": "pm25",
     "station": "Demo Station", "error": None},
    {"city": "Bengaluru", "aqi": 88, "category": "Satisfactory", "dominant_pollutant": "no2",
     "station": "Demo Station", "error": None},
    {"city": "Chennai", "aqi": 76, "category": "Satisfactory", "dominant_pollutant": "pm10",
     "station": "Demo Station", "error": None},
]


def demo_weather(hours: int = 72) -> List[Dict]:
    """Synthetic but pattern-realistic weather: calm/dry overnight, windier by afternoon."""
    out = []
    start = datetime.now().replace(minute=0, second=0, microsecond=0)
    for h in range(hours):
        t = start + timedelta(hours=h)
        hod = t.hour
        wind = 4 if hod < 7 else (16 if 13 <= hod <= 18 else 9)
        precip = 2.5 if (h > 48 and 13 <= hod <= 16) else 0.0
        out.append({
            "time": t.isoformat(timespec="minutes"),
            "temperature_c": 18 if hod < 7 else 27,
            "humidity_pct": 70 if hod < 7 else 40,
            "precipitation_mm": precip,
            "wind_speed_kmh": wind,
            "wind_direction_deg": 290,
        })
    return out