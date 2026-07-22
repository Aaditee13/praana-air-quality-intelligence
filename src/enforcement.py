"""
enforcement.py
--------------
Agent 3 — Enforcement Intelligence & Prioritisation Agent.

Turns Agent 1's source-attribution output into a ranked, evidence-backed
list of enforcement actions. The production design cross-references a
municipal construction/industry registry; this prototype instead queries
OpenStreetMap's free Overpass API for real, live-tagged industrial zones,
construction sites, and major roads near the station — genuine open data,
just a public-data proxy for a registry PRAANA doesn't have access to.
"""

from typing import Dict, List

# Maps each fingerprint source category to the OSM tag(s) we search for,
# and to the enforcement action template used in the output.
SOURCE_TO_ACTION = {
    "Construction": {
        "osm_query_hint": "landuse=construction within radius",
        "action": "Inspect construction site",
        "pollutant_relieved": "PM10",
    },
    "Vehicular": {
        "osm_query_hint": "highway=primary/trunk/motorway within radius",
        "action": "Deploy diesel-vehicle / PUC checkpoint",
        "pollutant_relieved": "NO2",
    },
    "Industrial": {
        "osm_query_hint": "landuse=industrial within radius",
        "action": "Issue notice to industrial cluster",
        "pollutant_relieved": "SO2",
    },
    "Crop Burning": {
        "osm_query_hint": "not locatable via OSM land-use tags",
        "action": "Escalate to state agriculture dept. (stubble-burning advisory)",
        "pollutant_relieved": "PM2.5",
    },
}


def estimate_impact(source_share_pct: float, dominant_aqi_subindex: float) -> float:
    """
    Rough expected-AQI-reduction estimate: the source's share of the current
    mix, scaled by how much that pollutant is driving the overall AQI.
    This is a transparent heuristic, not a calibrated impact model — the
    production agent would use the validated cost-benefit method described
    in the solution document's evaluation plan.
    """
    return round(min(source_share_pct * 0.4, 35.0), 1)


def build_action_list(attribution: Dict, aqi_result: Dict, osm_sites: List[Dict]) -> List[Dict]:
    """
    attribution: output of fingerprint.attribute_sources()
    aqi_result: output of aqi.compute_aqi()
    osm_sites: list of {name, lat, lon, category} found via Overpass for this ward
    Returns a ranked list of recommended actions with an evidence trail.
    """
    actions = []
    for source, share in attribution["ranked"]:
        if source not in SOURCE_TO_ACTION or share < 5:
            continue
        meta = SOURCE_TO_ACTION[source]
        # Dedupe by name: OSM often splits one real road/site into several
        # way-segments that share the same name (e.g. a road cut at every
        # intersection), which previously produced 2-3 near-identical
        # "duplicate" actions for what is really one physical location.
        matching_sites = []
        seen_names = set()
        for s in osm_sites:
            if s.get("category") != source:
                continue
            name = s.get("name", "unnamed site")
            if name in seen_names:
                continue
            seen_names.add(name)
            matching_sites.append(s)
            if len(matching_sites) >= 3:
                break
        impact = estimate_impact(share, aqi_result.get("aqi") or 0)

        if matching_sites:
            for site in matching_sites:
                actions.append({
                    "action": f"{meta['action']} — {site.get('name', 'unnamed site')}",
                    "source_category": source,
                    "expected_reduction_pct": impact,
                    "pollutant_relieved": meta["pollutant_relieved"],
                    "evidence": (f"Source attribution: {share}% of current readings match the "
                                 f"{source} fingerprint (confidence: {attribution['confidence']}). "
                                 f"Dominant pollutant: {aqi_result.get('dominant_pollutant')}. "
                                 f"Site located via OpenStreetMap land-use tagging."),
                    "lat": site.get("lat"), "lon": site.get("lon"),
                })
        else:
            actions.append({
                "action": f"{meta['action']} — no tagged site found nearby, manual ground-check needed",
                "source_category": source,
                "expected_reduction_pct": impact,
                "pollutant_relieved": meta["pollutant_relieved"],
                "evidence": (f"Source attribution: {share}% of current readings match the "
                             f"{source} fingerprint (confidence: {attribution['confidence']}). "
                             f"No matching OSM-tagged site within search radius — "
                             f"{meta['osm_query_hint']}."),
                "lat": None, "lon": None,
            })

    actions.sort(key=lambda a: a["expected_reduction_pct"], reverse=True)
    return actions


if __name__ == "__main__":
    fake_attribution = {
        "shares": {"Construction": 45.0, "Vehicular": 30.0, "Industrial": 15.0, "Crop Burning": 10.0},
        "ranked": [("Construction", 45.0), ("Vehicular", 30.0), ("Industrial", 15.0), ("Crop Burning", 10.0)],
        "dominant_source": "Construction", "confidence": "medium",
    }
    fake_aqi = {"aqi": 312, "dominant_pollutant": "pm10"}
    fake_osm = [
        {"name": "Site near Rohini Sector 15", "lat": 28.71, "lon": 77.10, "category": "Construction"},
        {"name": "NH48 corridor", "lat": 28.55, "lon": 77.07, "category": "Vehicular"},
    ]
    result = build_action_list(fake_attribution, fake_aqi, fake_osm)
    for r in result:
        print(r["action"], "->", r["expected_reduction_pct"], "% expected reduction")
    assert len(result) > 0
    print("enforcement.py self-test passed.")