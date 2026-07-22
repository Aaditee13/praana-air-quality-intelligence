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

# Public Overpass mirrors, tried in order if the primary times out or errors.
# overpass-api.de is the most commonly used free endpoint but is also the
# most frequently overloaded; falling through to alternates before giving
# up on live data reduces how often the app falls back to demo OSM sites.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

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

# aqi.py's CPCB breakpoint tables expect these units per pollutant.
# PM2.5/PM10/NO2/SO2/O3 in micrograms/m3, CO in milligrams/m3.
# This matters because OpenAQ sensors report CO in whatever unit that
# station's instrument uses (commonly ug/m3 or ppm), and silently feeding
# a ug/m3 CO reading into a mg/m3 table clamps the sub-index to 500
# ("Severe") for almost any real reading -- not a genuine severe-AQI
# result, just a units mismatch.
EXPECTED_UNITS = {
    "pm25": "µg/m³", "pm10": "µg/m³", "no2": "µg/m³", "so2": "µg/m³", "o3": "µg/m³",
    "co": "mg/m³",
}


def _normalize_pollutant_units(param: str, value: float, reported_units: Optional[str]) -> float:
    """
    Converts a raw OpenAQ sensor value into the unit aqi.py expects for that
    pollutant. If OpenAQ doesn't report units for this sensor, assumes the
    value is already in the expected unit (OpenAQ's default for most
    stations) rather than guessing.
    """
    if not reported_units:
        return value
    units = reported_units.strip().lower()

    if param == "co":
        if units in ("mg/m3", "mg/m³"):
            return value
        if units in ("µg/m3", "ug/m3", "µg/m³", "ug/m³"):
            return value / 1000.0  # ug/m3 -> mg/m3
        if units == "ppm":
            return value * 1.145  # ppm -> mg/m3 at standard temp/pressure (CO molar mass 28.01)
        return value

    # pm25/pm10/no2/so2/o3 all expect ug/m3
    if units in ("µg/m3", "ug/m3", "µg/m³", "ug/m³"):
        return value
    if units in ("mg/m3", "mg/m³"):
        return value * 1000.0  # mg/m3 -> ug/m3
    if units == "ppb" and param in ("no2", "so2", "o3"):
        # rough ppb -> ug/m3 conversion at standard conditions, molar-mass dependent;
        # good enough to avoid a gross unit-scale error, not lab-precise.
        molar_mass = {"no2": 46.0, "so2": 64.0, "o3": 48.0}[param]
        return value * molar_mass / 24.45
    return value


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
    sensors_by_id = {
        s["id"]: {
            "param": s.get("parameter", {}).get("name"),
            "units": (s.get("parameter", {}).get("units") or "").lower(),
        }
        for s in station.get("sensors", [])
    }
    for item in data.get("results", []):
        sensor_id = item.get("sensorsId") or item.get("sensorId")
        info = sensors_by_id.get(sensor_id, {})
        param = info.get("param")
        units = info.get("units")
        value = item.get("value")
        if param in OPENAQ_PARAMETERS and value is not None:
            readings[param] = _normalize_pollutant_units(param, value, units)

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
    last_error = None
    for mirror_url in OVERPASS_MIRRORS:
        try:
            resp = requests.post(
                mirror_url,
                data={"data": query},
                headers={"User-Agent": "PRAANA-AQI-Intelligence/1.0 (hackathon research project)"},
                timeout=REQUEST_TIMEOUT + 10,
            )
            resp.raise_for_status()
            data = resp.json()
            break  # success — stop trying further mirrors
        except requests.exceptions.RequestException as e:
            last_error = f"Overpass request to {mirror_url} failed: {e}"
            continue
        except ValueError as e:
            last_error = f"Overpass ({mirror_url}) returned invalid JSON: {e}"
            continue
    else:
        # every mirror in OVERPASS_MIRRORS failed
        return None, last_error or "All Overpass mirrors failed."

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