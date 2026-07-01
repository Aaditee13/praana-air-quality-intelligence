"""
fingerprint.py
--------------
Agent 1 — Geospatial Pollution Source Attribution Engine (lightweight version).

Real source apportionment (as described in the PRAANA solution document) uses
a model trained against IIT Kanpur / CPCB emission-inventory ground truth plus
a causal layer that adjusts for wind-driven transport. That requires the full
historical data pipeline this hackathon prototype doesn't have time to build.

This module implements a defensible, transparent v1: cosine-similarity
matching of a live pollutant mix against reference emission "fingerprints"
(relative pollutant ratios per source category, sourced from published
source-apportionment literature for Indian cities). It's a real, working
classifier — just a simpler one than the production design — and every
output exposes a confidence score, never disguising the heuristic as a
black box.
"""

import math
from typing import Dict, List, Tuple

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

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co"]


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
                 "fingerprints — a lightweight stand-in for the causal-AI "
                 "classifier described in the full solution document."),
    }


if __name__ == "__main__":
    sample = {"pm25": 180, "pm10": 220, "no2": 85, "so2": 12, "co": 45}
    result = attribute_sources(sample)
    print("Sample reading:", sample)
    print("Attribution:", result)
    assert abs(sum(result["shares"].values()) - 100) < 0.5
    print("fingerprint.py self-test passed.")
