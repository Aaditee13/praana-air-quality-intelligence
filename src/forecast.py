"""
forecast.py
-----------
Agent 2 — Hyperlocal Predictive AQI Forecasting Agent (lightweight version).

The production design (see solution document) is a Graph Neural Network
across the sensor network blended with a physics-based atmospheric
dispersion model, trained on five years of history. That needs a historical
data pipeline this hackathon prototype doesn't have time to assemble.

This module is a real, working stand-in: a persistence baseline adjusted by
a transparent meteorological heuristic (low wind + no rain traps pollution;
high wind + rain disperses it), plus a simple diurnal traffic-pattern bump.
It consumes real Open-Meteo hourly forecast data. Every forecast point ships
with an explicit confidence band so the simplification is never hidden.
"""

from datetime import datetime
from typing import Dict, List, Optional


def _wind_factor(wind_speed_kmh: float) -> float:
    """Low wind traps pollution (+), high wind disperses it (-). Clamped to ±0.30."""
    factor = 0.30 - (wind_speed_kmh / 20.0) * 0.30
    return max(-0.30, min(0.30, factor))


def _rain_factor(precip_mm: float) -> float:
    """Rain washes pollution out of the air. Clamped to -0.40 .. 0."""
    factor = -0.40 * min(precip_mm, 10.0) / 10.0
    return max(-0.40, factor)


def _diurnal_factor(hour_of_day: int) -> float:
    """Rush-hour and still-night bumps, based on the documented Delhi pattern."""
    if hour_of_day in (7, 8, 9, 10, 18, 19, 20, 21):
        return 0.12  # rush hour
    if hour_of_day in (0, 1, 2, 3, 4, 5):
        return 0.08  # calm night air, low boundary layer
    return 0.0


def forecast_aqi(current_aqi: float, hourly_weather: List[Dict], horizon_hours: int = 72) -> List[Dict]:
    """
    hourly_weather: list of dicts with keys 'time' (ISO string), 'wind_speed_kmh',
                     'precipitation_mm', sourced from the Open-Meteo hourly forecast.
                     Must have at least `horizon_hours` entries, ordered chronologically.
    Returns a list of {time, hour_offset, predicted_aqi, lower, upper} dicts.
    """
    points = []
    smoothed_multiplier = 0.0  # running smoother so the curve isn't jumpy

    for i, hour in enumerate(hourly_weather[:horizon_hours]):
        wind = float(hour.get("wind_speed_kmh", 10.0) or 10.0)
        precip = float(hour.get("precipitation_mm", 0.0) or 0.0)
        try:
            hour_of_day = datetime.fromisoformat(hour["time"]).hour
        except Exception:
            hour_of_day = i % 24

        raw_multiplier = _wind_factor(wind) + _rain_factor(precip) + _diurnal_factor(hour_of_day)
        # exponential smoothing so consecutive hours don't whipsaw
        smoothed_multiplier = 0.6 * smoothed_multiplier + 0.4 * raw_multiplier

        predicted = max(5.0, current_aqi * (1 + smoothed_multiplier))
        band = 0.10 + 0.05 * (i / max(horizon_hours, 1))  # uncertainty widens with horizon

        points.append({
            "time": hour.get("time"),
            "hour_offset": i,
            "predicted_aqi": round(predicted),
            "lower": round(predicted * (1 - band)),
            "upper": round(predicted * (1 + band)),
            "wind_speed_kmh": wind,
            "precipitation_mm": precip,
        })

    return points


def persistence_baseline(current_aqi: float, horizon_hours: int = 72) -> List[float]:
    """The naive 'tomorrow looks like today' baseline used in the evaluation plan."""
    return [round(current_aqi)] * horizon_hours


if __name__ == "__main__":
    # synthetic 24h of weather: calm + dry overnight, windy + rainy by afternoon
    fake_weather = []
    for h in range(72):
        hod = h % 24
        fake_weather.append({
            "time": f"2026-06-30T{hod:02d}:00",
            "wind_speed_kmh": 5 if hod < 12 else 18,
            "precipitation_mm": 0 if hod < 14 else 3,
        })
    fc = forecast_aqi(210, fake_weather, horizon_hours=72)
    print("First 3 forecast points:", fc[:3])
    print("Last forecast point:", fc[-1])
    assert len(fc) == 72
    assert all(p["lower"] <= p["predicted_aqi"] <= p["upper"] for p in fc)
    print("forecast.py self-test passed.")