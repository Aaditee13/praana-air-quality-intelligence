"""
streamlit_app.py
-----------------
PRAANA dashboard — the working prototype tying all five agents together.

Run with:
    streamlit run app/streamlit_app.py

Needs (see ../requirements.txt): streamlit, requests, pandas, plotly, python-dotenv
For live data, set OPENAQ_API_KEY in a .env file at the project root
(free key: https://explore.openaq.org/register). Open-Meteo and OSM Overpass
need no key. "Demo data" mode works with no setup at all.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# allow `import src.xxx` whether this is run from repo root or app/ folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import advisory, aqi, data_sources as ds, demo_data, enforcement, fingerprint, forecast, multicity

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; OPENAQ_API_KEY can also be set as a real env var

st.set_page_config(page_title="PRAANA — Air Quality Intelligence", layout="wide", page_icon="\U0001F32B\uFE0F")

AQI_CATEGORY_COLORS = {
    "Good": "#00A86B", "Satisfactory": "#6FCF97", "Moderate": "#F2C94C",
    "Poor": "#F2994A", "Very Poor": "#EB5757", "Severe": "#8B2942", "Unknown": "#888888",
}

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## PRAANA")
    st.caption("Predictive Real-time Attribution and Action for National Air quality")
    st.divider()

    city = st.selectbox("City", list(ds.CITIES.keys()), index=0)
    ward_label = st.text_input("Ward / area label (for the advisory text)", value="Rohini")

    data_mode = st.radio(
        "Data source",
        ["Live data (real APIs)", "Demo data (offline, no setup)"],
        index=1,
        help="Live mode calls OpenAQ, Open-Meteo, and OSM Overpass in real time. "
             "Demo mode uses fixed sample data so the app always works, even offline.",
    )
    radius_km = st.slider("Search radius (km)", 5, 50, 25)
    language = st.selectbox("Advisory language", list(advisory.LANGUAGE_TEMPLATES.keys()))

    st.divider()
    if data_mode.startswith("Live") and not ds.get_openaq_api_key():
        st.warning("No OPENAQ_API_KEY found. Pollutant readings will fail until you "
                   "add one to your .env file. Weather and OSM data don't need a key.")
    st.caption("Free API signup: [OpenAQ](https://explore.openaq.org/register)")

coords = ds.CITIES[city]
is_live = data_mode.startswith("Live")

# ---------------------------------------------------------------------------
# Fetch data (live or demo) — cached per (city, mode, radius) for the session
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_readings(lat, lon, radius_m, live: bool):
    if not live:
        return demo_data.DEMO_READINGS, None
    readings, err = ds.fetch_latest_readings(lat, lon, radius_m=radius_m)
    return readings, err


@st.cache_data(ttl=600, show_spinner=False)
def load_weather(lat, lon, live: bool):
    if not live:
        return demo_data.demo_weather(72), None
    weather, err = ds.fetch_weather_forecast(lat, lon, hours=72)
    return weather, err


@st.cache_data(ttl=600, show_spinner=False)
def load_osm(lat, lon, radius_m, live: bool):
    if not live:
        return demo_data.DEMO_OSM_SITES, None
    sites, err = ds.fetch_osm_landuse(lat, lon, radius_m=min(radius_m * 1000, 5000))
    return sites, err


with st.spinner("Pulling live data..." if is_live else "Loading demo data..."):
    readings, readings_err = load_readings(coords["lat"], coords["lon"], radius_km * 1000, is_live)
    weather, weather_err = load_weather(coords["lat"], coords["lon"], is_live)
    osm_sites, osm_err = load_osm(coords["lat"], coords["lon"], radius_km, is_live)

# graceful fallback: if live readings/weather failed, drop to demo data for that
# piece specifically so the rest of the dashboard still works during a demo
using_fallback = []
if readings_err:
    readings = demo_data.DEMO_READINGS
    using_fallback.append(f"pollutant readings ({readings_err})")
if weather_err:
    weather = demo_data.demo_weather(72)
    using_fallback.append(f"weather forecast ({weather_err})")
if osm_err:
    osm_sites = demo_data.DEMO_OSM_SITES
    using_fallback.append(f"OSM land-use ({osm_err})")

if using_fallback:
    with st.expander("\u26A0\uFE0F Some live data sources were unavailable — showing sample data instead", expanded=False):
        for msg in using_fallback:
            st.write("-", msg)

# ---------------------------------------------------------------------------
# Run the agents
# ---------------------------------------------------------------------------
clean_readings = {k: v for k, v in readings.items() if not str(k).startswith("_")}
aqi_result = aqi.compute_aqi(clean_readings)
attribution = fingerprint.attribute_sources(clean_readings)
forecast_points = forecast.forecast_aqi(aqi_result["aqi"] or 150, weather, horizon_hours=72)
baseline_points = forecast.persistence_baseline(aqi_result["aqi"] or 150, horizon_hours=72)
action_list = enforcement.build_action_list(attribution, aqi_result, osm_sites)
tomorrow_aqi = forecast_points[23]["predicted_aqi"] if len(forecast_points) > 23 else aqi_result["aqi"]
advisory_out = advisory.generate_advisory(ward_label or city, aqi_result["aqi"] or 150, tomorrow_aqi, language)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("PRAANA \u2014 Urban Air Quality Intelligence")
st.caption(f"{city} \u00b7 station: {readings.get('_station_name', 'n/a')} \u00b7 "
           f"data mode: {'live' if is_live and not using_fallback else 'demo / fallback'}")

col1, col2, col3, col4 = st.columns(4)
color = AQI_CATEGORY_COLORS.get(aqi_result["category"], "#888888")
with col1:
    st.markdown(f"""<div style="background-color:{color}22;border-radius:10px;padding:14px;">
        <span style="color:{color};font-size:14px;font-weight:600;">CURRENT AQI</span><br>
        <span style="font-size:38px;font-weight:700;color:#1B3B5F;">{aqi_result['aqi']}</span>
        <span style="font-size:16px;color:{color};font-weight:600;"> {aqi_result['category']}</span>
        </div>""", unsafe_allow_html=True)
with col2:
    st.metric("Dominant pollutant", (aqi_result["dominant_pollutant"] or "n/a").upper())
with col3:
    st.metric("Forecast (+24h)", tomorrow_aqi)
with col4:
    st.metric("Top attributed source", attribution["dominant_source"])

st.divider()

# ---------------------------------------------------------------------------
# Tabs — one per agent
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Agent 1 \u00b7 Source Attribution",
    "Agent 2 \u00b7 Forecast",
    "Agent 3 \u00b7 Enforcement",
    "Agent 4 \u00b7 Multi-City",
    "Agent 5 \u00b7 Citizen Advisory",
])

with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Who is causing it right now")
        fig = px.pie(
            names=list(attribution["shares"].keys()),
            values=list(attribution["shares"].values()),
            color_discrete_sequence=["#1B3B5F", "#4F86A0", "#E8862E", "#8FB7C9"],
            hole=0.35,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Confidence: **{attribution['confidence']}** \u00b7 {attribution['note']}")
    with right:
        st.subheader("Live reading used")
        st.json({k: v for k, v in clean_readings.items()})
        st.subheader("CPCB sub-indices")
        st.dataframe(pd.DataFrame(aqi_result["sub_indices"].items(), columns=["Pollutant", "Sub-index"]),
                     width='stretch', hide_index=True)

with tab2:
    st.subheader("Hyperlocal AQI forecast \u2014 next 72 hours")
    df = pd.DataFrame(forecast_points)
    df["persistence_baseline"] = baseline_points[:len(df)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["hour_offset"], y=df["upper"], line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["hour_offset"], y=df["lower"], line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(232,134,46,0.15)",
                              name="confidence band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["hour_offset"], y=df["predicted_aqi"], mode="lines",
                              line=dict(color="#E8862E", width=3), name="Predicted AQI"))
    fig.add_trace(go.Scatter(x=df["hour_offset"], y=df["persistence_baseline"], mode="lines",
                              line=dict(color="#5C6B73", width=2, dash="dash"), name="Persistence baseline"))
    fig.update_layout(xaxis_title="Hours from now", yaxis_title="AQI", height=420,
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width='stretch')
    st.caption("Heuristic forecast: persistence baseline adjusted for wind, rain, and a rush-hour/"
               "still-night diurnal pattern from live Open-Meteo data. Production design (see solution "
               "document) uses a GNN blended with atmospheric dispersion modelling.")

with tab3:
    st.subheader("Ranked enforcement actions")
    if not action_list:
        st.info("No actions ranked \u2014 source shares were all below the reporting threshold.")
    else:
        for i, a in enumerate(action_list, start=1):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**{i}. {a['action']}**")
                    st.caption(a["evidence"])
                with c2:
                    st.metric("Expected impact", f"{a['expected_reduction_pct']}%")
    st.caption("Sites located via live OpenStreetMap Overpass queries for construction/industrial "
               "land-use and major roads near the selected city centre.")

with tab4:
    st.subheader("Cross-city comparison")
    if st.button("Run multi-city comparison", type="primary"):
        with st.spinner("Comparing cities..." if is_live else "Loading demo comparison..."):
            rows = multicity.compare_cities(radius_m=radius_km * 1000) if is_live else demo_data.DEMO_MULTICITY
        st.session_state["multicity_rows"] = rows

    rows = st.session_state.get("multicity_rows", demo_data.DEMO_MULTICITY if not is_live else None)
    if rows:
        df_mc = pd.DataFrame(rows)
        plot_df = df_mc.dropna(subset=["aqi"])
        if not plot_df.empty:
            fig = px.bar(plot_df, x="city", y="aqi", color="category",
                         color_discrete_map=AQI_CATEGORY_COLORS, text="aqi")
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, width='stretch')
        st.dataframe(df_mc, width='stretch', hide_index=True)
    else:
        st.info("Click the button above to fetch a live AQI comparison across all five cities.")

with tab5:
    st.subheader(f"Citizen advisory \u2014 {language}")
    st.info(advisory_out["text"])
    groups = advisory.vulnerable_groups_flag(aqi_result["aqi"] or 0)
    if groups:
        st.warning("Flagged vulnerable groups: " + ", ".join(groups))
    else:
        st.success("No vulnerable groups flagged at this AQI level.")
