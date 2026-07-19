"""
fingerprint.py
--------------
Agent 1 — Geospatial Pollution Source Attribution Engine.

v1 (attribute_sources): cosine-similarity matching of a live pollutant mix
against reference emission "fingerprints" (relative pollutant ratios per
source category, sourced from published source-apportionment literature
for Indian cities). Unchanged from before.

v2 (attribute_sources_with_wind) — NEW: a real, working wind-confounder
adjustment. The solution document (Section 6.3) describes exactly this
scenario: "If wind is blowing from the northwest at 15 km/h and a
construction site sits 2km northwest of a ward where PM10 spikes, the
causal model can confirm the construction site as the responsible source
rather than misattributing the spike to local traffic." Previously that
was prose only. This function actually does it:

  1. Take the chemical-fingerprint shares from attribute_sources() as a
     prior.
  2. For every nearby land-use site (from OSM Overpass data — see
     data_sources.fetch_osm_landuse), compute the compass bearing from
     the sensor to that site.
  3. Compare that bearing to the live wind direction. A site is "upwind"
     (plausibly contributing right now) if wind is blowing FROM roughly
     that direction TOWARD the sensor.
  4. Sites that are upwind add positive evidence for their source
     category; sites that are downwind (their emissions are blowing
     away from the sensor, not toward it) are discounted.
  5. Blend that geospatial evidence with the chemical-fingerprint prior,
     and raise or lower confidence depending on whether the two lines of
     evidence agree or conflict.

This is still a heuristic, not a full DoWhy/EconML counterfactual causal
graph (that remains the documented production roadmap item) — but it is
a genuine confounder adjustment using real wind + real land-use geometry,
not just a chemical ratio match.
"""

import math
from typing import Dict, List, Optional, Tuple

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co"]

# Reference fingerprints: relative contribution shares per pollutant, per
# source category. These are illustrative ratios consistent with published
# Indian source-apportionment studies (e.g. IIT Kanpur's Delhi SAFAR/SOURCE
# work) — for a production system these would be re-estimated per city from
# the official emission inventory rather than hard-coded.
FINGERPRINTS: Dict[str, Dict[str, float]] = {
    "Vehicular":     {"pm25": 0.30, "pm10": 0.15, "no2": 0.35, "so2": 0.05, "co": 0.15},
    "Construction":  {"pm25": 0.15, "pm10": 0.55, "no2": 0.05, "so2": 0.05, "co": 0.20},
    "Crop Burning":  {"pm25": 0.45, "pm10": 0.20, "no2": 0.05, "so2": 0.05, "co": 0.25},
    "Industrial":    {"pm25": 0.20, "pm10": 0.15, "no2": 0.25, "so2": 0.30, "co": 0.10},
}

# Maps OSM land-use categories (from data_sources.fetch_osm_landuse) onto
# the same source categories used by the chemical fingerprint library.
OSM_CATEGORY_TO_SOURCE = {
    "Construction": "Construction",
    "Industrial": "Industrial",
    "Vehicular": "Vehicular",
    # Crop Burning has no OSM land-use tag - it comes from MODIS fire
    # hotspots instead, handled the same way if/when those are passed in.
}

# How far off-axis (degrees) a site can be from the wind bearing and still
# count as "upwind." 45 degrees is a reasonably generous cone - narrow
# enough to be meaningful, wide enough to tolerate GPS/wind noise.
UPWIND_CONE_DEG = 45.0


def _normalize(vec: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in vec.values() if v is not None and v > 0)
    if total <= 0:
        return {k: 0.0 for k in vec}
    return {k: (v / total if v else 0.0) for k, v in vec.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    num = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in POLLUTANTS)
    norm_a = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in POLLUTANTS))
    norm_b = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in POLLUTANTS))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return num / (norm_a * norm_b)


def attribute_sources(readings: Dict[str, float]) -> Dict:
    """
    readings: dict with raw concentrations for any subset of
              pm25, pm10, no2, so2, co.
    Returns source-share breakdown (sums to 100) and a confidence note.
    Unchanged from v1 - this is the chemical-fingerprint-only classifier.
    """
    clean = {k: float(readings.get(k) or 0.0) for k in POLLUTANTS}
    live_vec = _normalize(clean)

    sims: Dict[str, float] = {}
    for source, fp in FINGERPRINTS.items():
        sims[source] = max(_cosine(live_vec, fp), 0.0)

    total_sim = sum(sims.values())
    if total_sim == 0:
        n = len(FINGERPRINTS)
        shares = {k: round(100 / n, 1) for k in FINGERPRINTS}
        confidence = "low"
    else:
        shares = {k: round((v / total_sim) * 100, 1) for k, v in sims.items()}
        top_two = sorted(shares.values(), reverse=True)[:2]
        gap = top_two[0] - top_two[1] if len(top_two) > 1 else top_two[0]
        confidence = "high" if gap > 25 else ("medium" if gap > 10 else "low")

    ranked: List[Tuple[str, float]] = sorted(shares.items(), key=lambda x: x[1], reverse=True)

    return {
        "shares": shares,
        "ranked": ranked,
        "dominant_source": ranked[0][0],
        "confidence": confidence,
        "pollutants_used": [k for k, v in clean.items() if v > 0],
        "note": ("Heuristic cosine-similarity match against reference emission "
                 "fingerprints - chemical evidence only, no wind adjustment."),
    }


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing (0-360, 0=North) from point 1 to point 2."""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angular_diff(a: float, b: float) -> float:
    """Smallest difference between two compass bearings, 0-180."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _upwind_score(sensor_lat: float, sensor_lon: float, site_lat: float, site_lon: float,
                   wind_from_deg: float) -> float:
    """
    Returns 1.0 if the site is directly upwind of the sensor (its emissions
    are blowing straight toward the sensor right now), fading linearly to
    0.0 at the edge of the upwind cone, and 0.0 beyond it (including
    anywhere downwind).

    Meteorological convention: wind_from_deg is the direction the wind is
    COMING FROM. For a site's emissions to reach the sensor, the wind must
    blow from the site's direction toward the sensor - i.e. the compass
    bearing from the sensor to the site should roughly match wind_from_deg.
    """
    bearing_to_site = _bearing_deg(sensor_lat, sensor_lon, site_lat, site_lon)
    diff = _angular_diff(bearing_to_site, wind_from_deg)
    if diff >= UPWIND_CONE_DEG:
        return 0.0
    return 1.0 - (diff / UPWIND_CONE_DEG)


def attribute_sources_with_wind(
    readings: Dict[str, float],
    sensor_lat: float,
    sensor_lon: float,
    wind_from_deg: float,
    nearby_sites: Optional[List[Dict]] = None,
    chemical_weight: float = 0.6,
) -> Dict:
    """
    Wind-adjusted source attribution. This is the real, working version of
    the causal-adjustment idea described in Section 6.3 of the solution
    document.

    Args:
        readings: pollutant concentrations, same as attribute_sources().
        sensor_lat, sensor_lon: coordinates of the monitoring station /
            ward centroid whose reading we're attributing.
        wind_from_deg: current wind direction in compass degrees, the
            direction the wind is blowing FROM (0=N, 90=E, 180=S, 270=W).
            This is exactly what Open-Meteo's `wind_direction_10m` field
            returns (see data_sources.fetch_weather_forecast).
        nearby_sites: list of {"name", "lat", "lon", "category"} dicts,
            e.g. the output of data_sources.fetch_osm_landuse(). category
            should be one of "Construction", "Industrial", "Vehicular".
            If None or empty, this function returns the same result as
            plain attribute_sources() (no geospatial evidence available).
        chemical_weight: how much weight the chemical fingerprint keeps
            relative to the wind/geospatial evidence when both are
            available. 0.6 means 60% chemical, 40% wind-adjusted.

    Returns everything attribute_sources() returns, plus:
        wind_evidence: per-category upwind-site count and strength
        agreement: "confirmed" | "conflict" | "no_geospatial_evidence"
    """
    base = attribute_sources(readings)

    if not nearby_sites:
        base["wind_evidence"] = {}
        base["agreement"] = "no_geospatial_evidence"
        base["note"] = base["note"] + " No nearby OSM sites supplied, so no wind adjustment was applied."
        return base

    # Aggregate upwind evidence per source category.
    wind_scores: Dict[str, float] = {src: 0.0 for src in FINGERPRINTS}
    site_detail = []
    for site in nearby_sites:
        source = OSM_CATEGORY_TO_SOURCE.get(site.get("category"))
        if source is None:
            continue
        score = _upwind_score(sensor_lat, sensor_lon, site["lat"], site["lon"], wind_from_deg)
        if score > 0:
            wind_scores[source] += score
            site_detail.append({
                "name": site.get("name", "Unnamed site"),
                "category": source,
                "upwind_score": round(score, 2),
            })

    total_wind = sum(wind_scores.values())
    if total_wind == 0:
        # Sites exist nearby, but none of them are upwind right now -
        # geospatially, none of them can be responsible for this reading.
        base["wind_evidence"] = {"sites_checked": len(nearby_sites), "upwind_sites": []}
        base["agreement"] = "no_upwind_sources"
        base["note"] = (base["note"] + " " +
                         f"{len(nearby_sites)} nearby site(s) checked against wind direction "
                         f"{wind_from_deg:.0f}°; none are currently upwind, so geospatial "
                         "evidence neither confirms nor overrides the chemical read.")
        return base

    wind_shares = {src: (v / total_wind) * 100 for src, v in wind_scores.items()}

    # Blend chemical fingerprint (prior) with wind/geospatial evidence.
    blended = {
        src: chemical_weight * base["shares"].get(src, 0.0) + (1 - chemical_weight) * wind_shares.get(src, 0.0)
        for src in FINGERPRINTS
    }
    blend_total = sum(blended.values()) or 1.0
    blended = {src: round((v / blend_total) * 100, 1) for src, v in blended.items()}
    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)

    top_two = sorted(blended.values(), reverse=True)[:2]
    gap = top_two[0] - top_two[1] if len(top_two) > 1 else top_two[0]

    chemical_top = base["dominant_source"]
    wind_top = max(wind_shares, key=wind_shares.get)
    if chemical_top == wind_top:
        agreement = "confirmed"
        confidence = "high" if gap > 20 else "medium"
    else:
        agreement = "conflict"
        confidence = "low"

    base["shares"] = blended
    base["ranked"] = ranked
    base["dominant_source"] = ranked[0][0]
    base["confidence"] = confidence
    base["agreement"] = agreement
    base["wind_evidence"] = {
        "sites_checked": len(nearby_sites),
        "upwind_sites": site_detail,
        "wind_dominant_source": wind_top,
        "chemical_dominant_source": chemical_top,
    }
    base["note"] = (
        f"Chemical fingerprint pointed to {chemical_top}; wind/geospatial evidence "
        f"({len(site_detail)} upwind site(s), wind from {wind_from_deg:.0f}°) points to "
        f"{wind_top}. Evidence {agreement} - blended {int(chemical_weight*100)}% chemical / "
        f"{int((1-chemical_weight)*100)}% geospatial."
    )
    return base


if __name__ == "__main__":
    sample = {"pm25": 180, "pm10": 220, "no2": 85, "so2": 12, "co": 45}

    print("--- v1: chemical fingerprint only ---")
    result = attribute_sources(sample)
    print("Attribution:", result["ranked"], "| confidence:", result["confidence"])
    assert abs(sum(result["shares"].values()) - 100) < 0.5

    print("\n--- v2: wind-adjusted, matching the Section 6.3 example ---")
    # Sensor at a Delhi ward centroid. Wind blowing FROM the northwest
    # (315 degrees) at 15 km/h. A construction site sits ~2km northwest
    # of the sensor - exactly upwind - so its emissions should be
    # confirmed as reaching the sensor right now.
    sensor_lat, sensor_lon = 28.7041, 77.1025
    # ~2km to the northwest of the sensor:
    site_lat, site_lon = 28.7169, 77.0868
    wind_from_deg = 315.0

    nearby_sites = [
        {"name": "Rohini Sector 15 construction site", "lat": site_lat, "lon": site_lon, "category": "Construction"},
        {"name": "NH48 corridor", "lat": 28.70, "lon": 77.12, "category": "Vehicular"},  # roughly SE, downwind
    ]

    wind_result = attribute_sources_with_wind(
        sample, sensor_lat, sensor_lon, wind_from_deg, nearby_sites
    )
    print("Blended attribution:", wind_result["ranked"])
    print("Agreement:", wind_result["agreement"], "| confidence:", wind_result["confidence"])
    print("Note:", wind_result["note"])
    assert abs(sum(wind_result["shares"].values()) - 100) < 0.5
    assert wind_result["wind_evidence"]["upwind_sites"], "expected the NW construction site to register as upwind"

    print("\nfingerprint.py self-test passed.")