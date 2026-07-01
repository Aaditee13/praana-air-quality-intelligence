"""
multicity.py
------------
Agent 4 — Multi-City Comparative Intelligence Dashboard.

Same pipeline, any city: this module just loops the live OpenAQ + AQI
pipeline across PRAANA's reference city list. Real production version would
also track intervention-effectiveness over time (Section 9 of the solution
document); that needs a persistence layer (database) this prototype doesn't
have, so it's left as a documented next step rather than faked.
"""

from typing import Dict, List

from . import data_sources as ds
from .aqi import compute_aqi


def compare_cities(radius_m: int = 25000) -> List[Dict]:
    """
    Returns one row per city in data_sources.CITIES with live AQI if
    reachable, or a clear per-city error if not (e.g. no nearby station).
    """
    rows = []
    for city, coords in ds.CITIES.items():
        readings, err = ds.fetch_latest_readings(coords["lat"], coords["lon"], radius_m=radius_m)
        if err:
            rows.append({"city": city, "aqi": None, "category": None,
                         "station": None, "error": err})
            continue
        clean = {k: v for k, v in readings.items() if not k.startswith("_")}
        aqi_result = compute_aqi(clean)
        rows.append({
            "city": city,
            "aqi": aqi_result["aqi"],
            "category": aqi_result["category"],
            "dominant_pollutant": aqi_result["dominant_pollutant"],
            "station": readings.get("_station_name"),
            "error": None,
        })
    return rows


if __name__ == "__main__":
    print("compare_cities() makes live OpenAQ calls per city — run on your own")
    print("machine with OPENAQ_API_KEY set. Cities covered:", list(ds.CITIES.keys()))
