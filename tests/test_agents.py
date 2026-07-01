"""
tests/test_agents.py
---------------------
Unit tests for the agent logic that doesn't depend on live network calls
(aqi, fingerprint, forecast, enforcement, advisory). data_sources.py is
exercised separately, on a machine with internet access, since it makes
real calls to OpenAQ / Open-Meteo / Overpass.

Run with: pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import advisory, aqi, enforcement, fingerprint, forecast


def test_aqi_basic():
    result = aqi.compute_aqi({"pm25": 180, "pm10": 220, "no2": 85, "so2": 12, "co": 2.1})
    assert result["aqi"] is not None
    assert result["category"] in {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}
    assert result["dominant_pollutant"] in {"pm25", "pm10", "no2", "so2", "co"}


def test_aqi_empty_readings():
    result = aqi.compute_aqi({})
    assert result["aqi"] is None
    assert result["category"] == "Unknown"


def test_aqi_sub_index_clamps_out_of_range():
    # absurdly high concentration should clamp, not raise or return None
    si = aqi.sub_index("pm25", 99999)
    assert si is not None
    assert si <= 500


def test_fingerprint_shares_sum_to_100():
    result = fingerprint.attribute_sources({"pm25": 180, "pm10": 220, "no2": 85, "so2": 12, "co": 45})
    assert abs(sum(result["shares"].values()) - 100) < 0.5
    assert result["dominant_source"] in fingerprint.FINGERPRINTS


def test_fingerprint_handles_zero_readings():
    result = fingerprint.attribute_sources({})
    assert result["confidence"] == "low"
    assert abs(sum(result["shares"].values()) - 100) < 0.5


def test_forecast_band_contains_prediction():
    weather = [{"time": f"2026-06-30T{(h % 24):02d}:00", "wind_speed_kmh": 10, "precipitation_mm": 0}
               for h in range(72)]
    points = forecast.forecast_aqi(200, weather, horizon_hours=72)
    assert len(points) == 72
    assert all(p["lower"] <= p["predicted_aqi"] <= p["upper"] for p in points)


def test_forecast_rain_lowers_prediction():
    dry = [{"time": "2026-06-30T12:00", "wind_speed_kmh": 5, "precipitation_mm": 0}] * 24
    wet = [{"time": "2026-06-30T12:00", "wind_speed_kmh": 5, "precipitation_mm": 10}] * 24
    dry_forecast = forecast.forecast_aqi(200, dry, horizon_hours=24)[-1]["predicted_aqi"]
    wet_forecast = forecast.forecast_aqi(200, wet, horizon_hours=24)[-1]["predicted_aqi"]
    assert wet_forecast < dry_forecast


def test_enforcement_ranks_by_impact_descending():
    attribution = {
        "ranked": [("Construction", 45.0), ("Vehicular", 30.0), ("Industrial", 15.0), ("Crop Burning", 10.0)],
        "confidence": "medium",
    }
    aqi_result = {"aqi": 312, "dominant_pollutant": "pm10"}
    osm_sites = [{"name": "Test site", "lat": 28.7, "lon": 77.1, "category": "Construction"}]
    actions = enforcement.build_action_list(attribution, aqi_result, osm_sites)
    impacts = [a["expected_reduction_pct"] for a in actions]
    assert impacts == sorted(impacts, reverse=True)


def test_advisory_severity_matches_category():
    out = advisory.generate_advisory("TestWard", 312, 245, "English")
    assert out["severity"] == "Very Poor"
    assert "TestWard" in out["text"]


def test_advisory_hindi_renders():
    out = advisory.generate_advisory("TestWard", 312, 245, "Hindi")
    assert len(out["text"]) > 0
    assert out["language"] == "Hindi"


def test_vulnerable_groups_scale_with_severity():
    assert advisory.vulnerable_groups_flag(40) == []
    assert len(advisory.vulnerable_groups_flag(150)) == 3
    assert len(advisory.vulnerable_groups_flag(350)) == 5
