"""
data_sources.py
----------------
Live data fetchers for PRAANA's free, no-licence-cost data sources:

- Open-Meteo  : weather forecast (wind, rain, temp) — no API key required
- OpenAQ v3   : ground-station pollutant readings — free API key required
                (sign up at https://explore.openaq.org/register)
- OSM Overpass: industrial/construction/highway land-use near a point — no key

Every function fails soft: on network error, bad key, or no nearby stations,
it returns (None, error_message) instead of raising, so the Streamlit app
can show a clear status message instead of crashing. This also means these
functions cannot be exercised in a network-locked sandbox — they're written
against the documented API contracts and are meant to be run on your own
machine with internet access. Test them locally before demo day.
"""

import os
from typing import Dict, List, Optional, Tuple

import requests

OPENAQ_BASE = "https://api.openaq.org/v3"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
OVERPASS_BASE = "https://overpass-api.de/api/interpreter"

REQUEST_TIMEOUT = 15  # seconds

# Reference coordinates for the multi-city comparator (Agent 4)
CITIES = {
    "Delhi NCR":  {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":     {"lat": 19.0760, "lon": 72.8777},
    "Kolkata":    {"lat": 22.5726, "lon": 88.3639},
    "Bengaluru":  {"lat": 12.9716, "lon": 77.5946},
    "Chennai":    {"lat": 13.0827, "lon": 80.2707},
}

OPENAQ_PARAMETERS = ["pm25", "pm10", "no2", "so2", "co", "o3"]


def get_openaq_api_key() -> Optional[str]:
    return os.environ.get("OPENAQ_API_KEY")


# ---------------------------------------------------------------------------
# Open-Meteo — weather forecast (no key needed)
# ---------------------------------------------------------------------------
def fetch_weather_forecast(lat: float, lon: float, hours: int = 72) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """Returns hourly weather for the next `hours` hours, or (None, error)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
        "forecast_days": max(2, (hours // 24) + 1),
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    try:
        resp = requests.get(OPEN_METEO_BASE, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return None, f"Open-Meteo request failed: {e}"
    except ValueError as e:
        return None, f"Open-Meteo returned invalid JSON: {e}"

    try:
        hourly = data["hourly"]
        times = hourly["time"]
        result = []
        for i in range(min(hours, len(times))):
            result.append({
                "time": times[i],
                "temperature_c": hourly["temperature_2m"][i],
                "humidity_pct": hourly["relative_humidity_2m"][i],
                "precipitation_mm": hourly["precipitation"][i],
                "wind_speed_kmh": hourly["wind_speed_10m"][i],
                "wind_direction_deg": hourly["wind_direction_10m"][i],
            })
        return result, None
    except (KeyError, IndexError) as e:
        return None, f"Open-Meteo response missing expected fields: {e}"


# ---------------------------------------------------------------------------
# OpenAQ v3 — live pollutant readings (free API key required)
# ---------------------------------------------------------------------------
def fetch_nearest_stations(lat: float, lon: float, radius_m: int = 25000, limit: int = 5
                            ) -> Tuple[Optional[List[Dict]], Optional[str]]:
    api_key = get_openaq_api_key()
    if not api_key:
        return None, ("No OPENAQ_API_KEY set. Get a free key at "
                       "https://explore.openaq.org/register and put it in your .env file.")
    headers = {"X-API-Key": api_key}
    params = {"coordinates": f"{lat},{lon}", "radius": radius_m, "limit": limit}
    try:
        resp = requests.get(f"{OPENAQ_BASE}/locations", headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return None, f"OpenAQ /locations request failed: {e}"

    stations = data.get("results", [])
    if not stations:
        return None, f"No OpenAQ stations found within {radius_m/1000:.0f}km of ({lat:.3f},{lon:.3f})."
    return stations, None


def fetch_latest_readings(lat: float, lon: float, radius_m: int = 25000
                           ) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Finds the nearest OpenAQ station and returns its latest readings as
    {pollutant_slug: concentration}. Returns (None, error) on any failure.
    """
    stations, err = fetch_nearest_stations(lat, lon, radius_m=radius_m, limit=1)
    if err:
        return None, err

    station = stations[0]
    station_id = station.get("id")
    api_key = get_openaq_api_key()
    headers = {"X-API-Key": api_key}

    try:
        resp = requests.get(f"{OPENAQ_BASE}/locations/{station_id}/latest",
                             headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return None, f"OpenAQ /latest request failed: {e}"

    readings = {}
    sensors_by_id = {s["id"]: s.get("parameter", {}).get("name") for s in station.get("sensors", [])}
    for item in data.get("results", []):
        sensor_id = item.get("sensorsId") or item.get("sensorId")
        param = sensors_by_id.get(sensor_id)
        value = item.get("value")
        if param in OPENAQ_PARAMETERS and value is not None:
            readings[param] = value

    if not readings:
        return None, f"Station '{station.get('name')}' returned no usable pollutant readings right now."

    readings["_station_name"] = station.get("name")
    readings["_station_lat"] = station.get("coordinates", {}).get("latitude")
    readings["_station_lon"] = station.get("coordinates", {}).get("longitude")
    return readings, None


# ---------------------------------------------------------------------------
# OSM Overpass — land-use context near a point (no key needed)
# ---------------------------------------------------------------------------
def fetch_osm_landuse(lat: float, lon: float, radius_m: int = 3000
                       ) -> Tuple[Optional[List[Dict]], Optional[str]]:
    query = f"""
    [out:json][timeout:25];
    (
      way["landuse"="construction"](around:{radius_m},{lat},{lon});
      way["landuse"="industrial"](around:{radius_m},{lat},{lon});
      way["highway"~"^(primary|trunk|motorway)$"](around:{radius_m},{lat},{lon});
    );
    out center 20;
    """
    try:
        resp = requests.post(
            OVERPASS_BASE,
            data={"data": query},
            headers={"User-Agent": "PRAANA-AQI-Intelligence/1.0 (hackathon research project)"},
            timeout=REQUEST_TIMEOUT + 10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return None, f"Overpass request failed: {e}"
    except ValueError as e:
        return None, f"Overpass returned invalid JSON: {e}"

    sites = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        center = el.get("center", {})
        if "lat" not in center or "lon" not in center:
            continue
        if tags.get("landuse") == "construction":
            category = "Construction"
        elif tags.get("landuse") == "industrial":
            category = "Industrial"
        elif tags.get("highway"):
            category = "Vehicular"
        else:
            continue
        sites.append({
            "name": tags.get("name", f"Unnamed {category.lower()} site"),
            "lat": center["lat"], "lon": center["lon"],
            "category": category,
        })
    return sites, None


if __name__ == "__main__":
    print("This module makes live calls to Open-Meteo, OpenAQ, and Overpass.")
    print("It cannot be tested in a network-restricted sandbox — run it on your")
    print("own machine with internet access, e.g.:")
    print("  python -c \"from src.data_sources import fetch_weather_forecast; "
          "print(fetch_weather_forecast(28.6139, 77.2090, hours=3))\"")
