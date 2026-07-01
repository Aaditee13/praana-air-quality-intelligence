"""
aqi.py
------
India CPCB National AQI calculation from raw pollutant concentrations.

The CPCB AQI is computed per-pollutant as a sub-index using linear
interpolation between published breakpoints, and the overall AQI for a
location/hour is the MAX of the available sub-indices (the "worst pollutant
drives the index" rule CPCB itself uses).

Reference breakpoints: CPCB National Air Quality Index (NAQI), 24-hr
averages for PM2.5/PM10/NO2/SO2, 8-hr average for CO and O3.
Units expected: PM2.5/PM10/NO2/SO2/O3 in micrograms/m3, CO in mg/m3.
"""

from typing import Dict, Optional, Tuple

# (conc_low, conc_high, aqi_low, aqi_high) per pollutant, in CPCB's published units
BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (251, 380, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
        (251, 350, 201, 300), (351, 430, 301, 400), (431, 510, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
        (181, 280, 201, 300), (281, 400, 301, 400), (401, 500, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
        (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2100, 401, 500),
    ],
    "co": [  # mg/m3
        (0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),
        (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
        (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500),
    ],
}

AQI_BUCKETS = [
    (0, 50, "Good", "00A86B"),
    (51, 100, "Satisfactory", "6FCF97"),
    (101, 200, "Moderate", "F2C94C"),
    (201, 300, "Poor", "F2994A"),
    (301, 400, "Very Poor", "EB5757"),
    (401, 500, "Severe", "8B2942"),
]


def sub_index(pollutant: str, conc: Optional[float]) -> Optional[float]:
    """Linear-interpolated CPCB sub-index for one pollutant concentration."""
    if conc is None or conc < 0:
        return None
    table = BREAKPOINTS.get(pollutant.lower())
    if not table:
        return None
    # clamp to table bounds
    lo_bound = table[0][0]
    hi_bound = table[-1][1]
    c = max(lo_bound, min(conc, hi_bound))
    for c_lo, c_hi, i_lo, i_hi in table:
        if c_lo <= c <= c_hi:
            if c_hi == c_lo:
                return float(i_lo)
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (c - c_lo) + i_lo
    return None


def compute_aqi(readings: Dict[str, Optional[float]]) -> Dict:
    """
    readings: dict with any of keys pm25, pm10, no2, so2, co, o3 (raw concentrations).
    Returns overall AQI, the dominant pollutant, the category, and all sub-indices.
    """
    sub_indices = {}
    for pollutant, conc in readings.items():
        si = sub_index(pollutant, conc)
        if si is not None:
            sub_indices[pollutant] = round(si, 1)

    if not sub_indices:
        return {"aqi": None, "category": "Unknown", "color": "888888",
                "dominant_pollutant": None, "sub_indices": {}}

    dominant = max(sub_indices, key=sub_indices.get)
    aqi_val = sub_indices[dominant]
    category, color = "Unknown", "888888"
    for lo, hi, label, hexcolor in AQI_BUCKETS:
        if lo <= aqi_val <= hi:
            category, color = label, hexcolor
            break
    if aqi_val > 500:
        category, color = "Severe", "8B2942"

    return {
        "aqi": round(aqi_val),
        "category": category,
        "color": color,
        "dominant_pollutant": dominant,
        "sub_indices": sub_indices,
    }


if __name__ == "__main__":
    # quick self-test with a known-ish severe Delhi-winter-style reading
    sample = {"pm25": 180, "pm10": 220, "no2": 85, "so2": 12, "co": 2.1}
    result = compute_aqi(sample)
    print("Sample reading:", sample)
    print("Computed AQI:", result)
    assert result["aqi"] is not None
    assert result["dominant_pollutant"] in sample
    print("aqi.py self-test passed.")
