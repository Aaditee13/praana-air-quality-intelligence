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
import base64
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# allow `import src.xxx` whether this is run from repo root or app/ folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import advisory, aqi, data_sources as ds, demo_data, enforcement, fingerprint, forecast, forecast_model, multicity

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# st.set_page_config MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PRAANA — Air Quality Intelligence",
    layout="wide",
    page_icon="🌫️",
)

# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
def set_background(image_path):
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_img = base64.b64encode(img_data).decode()
    # detect format from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        * {{ font-family: 'Inter', sans-serif !important; }}

        /* Full-page background */
        .stApp {{
            background-image: url("data:{mime};base64,{b64_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Dark navy overlay so text stays readable */
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(4, 12, 30, 0.65);
            z-index: 0;
            pointer-events: none;
        }}

        /* Sidebar — dark panel */
        [data-testid="stSidebar"] {{
            background: rgba(4, 12, 30, 0.82) !important;
            border-right: 1px solid rgba(79, 134, 160, 0.20) !important;
        }}
        [data-testid="stSidebar"] * {{ color: #DCEAF2 !important; }}
        [data-testid="stSidebar"] .stSelectbox > div > div,
        [data-testid="stSidebar"] .stTextInput > div > div {{
            background: rgba(255,255,255,0.07) !important;
            border: 1px solid rgba(79,134,160,0.30) !important;
            border-radius: 6px !important;
            color: #DCEAF2 !important;
        }}

        /* Main block */
        .block-container {{
            background: transparent !important;
            padding-top: 1.2rem !important;
        }}

        /* Strip ALL individual element backgrounds and boxes */
        [data-testid="stVerticalBlock"] > div,
        [data-testid="stVerticalBlock"],
        div[data-testid="column"],
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stHorizontalBlock"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="metric-container"],
        [data-testid="stDataFrame"],
        .element-container,
        div.row-widget,
        div[class*="stMarkdown"] {{
            background: transparent !important;
            background-color: transparent !important;
            backdrop-filter: none !important;
            border: none !important;
            box-shadow: none !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background: transparent !important;
            border: none !important;
            border-bottom: 1px solid rgba(255,255,255,0.12) !important;
            padding: 0 !important;
            gap: 6px !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent !important;
            color: rgba(220,234,242,0.70) !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            border: none !important;
            padding: 10px 20px !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: transparent !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border-bottom: 2px solid #E8862E !important;
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            background: transparent !important;
            border: none !important;
            padding: 1.4rem 0 !important;
        }}

        /* Expander + alerts — subtle tint only */
        [data-testid="stExpander"] {{
            background: rgba(4, 12, 30, 0.45) !important;
            border: 1px solid rgba(232,134,46,0.25) !important;
            border-radius: 8px !important;
        }}
        .stAlert {{
            background: rgba(4, 12, 30, 0.45) !important;
            border: 1px solid rgba(79,134,160,0.20) !important;
            border-radius: 8px !important;
        }}

        /* Container with border */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(4, 12, 30, 0.35) !important;
            border: 1px solid rgba(79,134,160,0.18) !important;
            border-radius: 10px !important;
        }}

        /* Typography */
        h1 {{
            color: #FFFFFF !important;
            font-weight: 800 !important;
            font-size: 2.2rem !important;
            text-shadow: 0 2px 12px rgba(0,0,0,0.80) !important;
        }}
        h2, h3, h4 {{
            color: #DCEAF2 !important;
            font-weight: 600 !important;
            text-shadow: 0 1px 8px rgba(0,0,0,0.75) !important;
        }}
        p, li, div, span, label {{ color: #DCEAF2 !important; }}
        .stCaption {{ color: #7FA8BC !important; font-size: 11.5px !important; }}

        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: #FFFFFF !important;
            font-weight: 800 !important;
            font-size: 2rem !important;
            text-shadow: 0 2px 10px rgba(0,0,0,0.80) !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: #7FA8BC !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
        }}

        /* Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, #E8862E, #c96d1e) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            padding: 10px 24px !important;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, #c96d1e, #E8862E) !important;
            box-shadow: 0 4px 15px rgba(232,134,46,0.40) !important;
        }}

        /* Plotly fully transparent */
        .js-plotly-plot .plotly,
        .js-plotly-plot .plotly .plot-container,
        .js-plotly-plot .plotly .svg-container {{
            background: transparent !important;
        }}

        /* JSON viewer */
        .stJson,
        div[data-testid="stJson"] pre,
        div[data-testid="stJson"] code {{
            background: rgba(10, 25, 55, 0.38) !important;
            border-radius: 8px !important;
            backdrop-filter: blur(6px) !important;
        }}
        hr   {{ border-color: rgba(79,134,160,0.18) !important; }}
        code {{ background: rgba(255,255,255,0.08) !important; color: #DCEAF2 !important; }}
        [data-testid="stDataFrame"] > div,
        [data-testid="stDataFrame"] div[class*="stDataFrameResizable"],
        [data-testid="stDataFrame"] {{
            background: rgba(10, 25, 55, 0.38) !important;
            border-radius: 8px !important;
            backdrop-filter: blur(6px) !important;
            border: 1px solid rgba(79,134,160,0.15) !important;
        }}

        /* Hide Streamlit chrome */
        #MainMenu {{ visibility: hidden; }}

        /* Hide sidebar collapse button icon text */
        [data-testid="collapsedControl"] {{
        display: none !important;
        }}
        button[kind="header"] {{
        display: none !important;
        }}
        section[data-testid="stSidebar"] > div:first-child > div:first-child {{
        display: none !important;
        }}
        
        footer    {{ visibility: hidden; }}
        header    {{ background: transparent !important; box-shadow: none !important; }}
        </style>
    """, unsafe_allow_html=True)


# FIX: changed wallpaper.png → bg.jpeg to match your actual file name
BACKGROUND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "bg.jpeg"
)
set_background(BACKGROUND_PATH)

# ---------------------------------------------------------------------------
# AQI colour map
# ---------------------------------------------------------------------------
AQI_CATEGORY_COLORS = {
    "Good":         "#00A86B",
    "Satisfactory": "#6FCF97",
    "Moderate":     "#F2C94C",
    "Poor":         "#F2994A",
    "Very Poor":    "#EB5757",
    "Severe":       "#8B2942",
    "Unknown":      "#888888",
}

# ---------------------------------------------------------------------------
# Sidebar
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

coords  = ds.CITIES[city]
is_live = data_mode.startswith("Live")

# ---------------------------------------------------------------------------
# Fetch data — cached
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_readings(lat, lon, radius_m, live: bool):
    if not live:
        return demo_data.DEMO_READINGS, None
    return ds.fetch_latest_readings(lat, lon, radius_m=radius_m)


@st.cache_data(ttl=600, show_spinner=False)
def load_weather(lat, lon, live: bool):
    if not live:
        return demo_data.demo_weather(72), None
    return ds.fetch_weather_forecast(lat, lon, hours=72)


@st.cache_data(ttl=600, show_spinner=False)
def load_osm(lat, lon, radius_m, live: bool):
    if not live:
        return demo_data.DEMO_OSM_SITES, None
    return ds.fetch_osm_landuse(lat, lon, radius_m=min(radius_m * 1000, 5000))


# Real per-horizon GBR models (Section 13 RMSE 80.8/82.5/83.8), trained once
# and cached for the whole app's lifetime rather than retrained per request.
@st.cache_resource(show_spinner="Training validated per-horizon forecast models (one-time)...")
def get_trained_forecast_models():
    return forecast_model.train_all_horizon_models()


with st.spinner("Pulling live data..." if is_live else "Loading demo data..."):
    readings,  readings_err = load_readings(coords["lat"], coords["lon"], radius_km * 1000, is_live)
    weather,   weather_err  = load_weather(coords["lat"], coords["lon"], is_live)
    osm_sites, osm_err      = load_osm(coords["lat"], coords["lon"], radius_km, is_live)

using_fallback = []
if readings_err:
    readings  = demo_data.DEMO_READINGS;    using_fallback.append(f"pollutant readings ({readings_err})")
if weather_err:
    weather   = demo_data.demo_weather(72); using_fallback.append(f"weather forecast ({weather_err})")
if osm_err:
    osm_sites = demo_data.DEMO_OSM_SITES;   using_fallback.append(f"OSM land-use ({osm_err})")

if using_fallback:
    with st.expander("⚠️ Some live data sources were unavailable — showing sample data instead", expanded=False):
        for msg in using_fallback:
            st.write("-", msg)

# ---------------------------------------------------------------------------
# Run agents
# ---------------------------------------------------------------------------
clean_readings  = {k: v for k, v in readings.items() if not str(k).startswith("_")}
aqi_result      = aqi.compute_aqi(clean_readings)
attribution     = fingerprint.attribute_sources_with_wind(
    clean_readings,
    sensor_lat=coords["lat"],
    sensor_lon=coords["lon"],
    wind_from_deg=weather[0]["wind_direction_deg"] if weather else 270,
    nearby_sites=osm_sites,
)
forecast_points = forecast.forecast_aqi(aqi_result["aqi"] or 150, weather, horizon_hours=72)
baseline_points = forecast.persistence_baseline(aqi_result["aqi"] or 150, horizon_hours=72)

# Anchor the 24h/48h/72h checkpoints to the real trained per-horizon GBR
# models (the ones RMSE-validated in Section 13) instead of leaving the
# whole curve as only the lighter wind/rain heuristic. The heuristic still
# supplies the smooth hourly shape between checkpoints; the three marked
# checkpoints are genuine model output.
trained_forecast_models = get_trained_forecast_models()
weather_by_offset = {i: w for i, w in enumerate(weather)}
validated_checkpoints = forecast_model.predict_live(
    trained_forecast_models, aqi_result["aqi"] or 150, datetime.now(), weather_by_offset
)
for h in (24, 48, 72):
    idx = h - 1
    if idx < len(forecast_points):
        forecast_points[idx]["predicted_aqi"] = validated_checkpoints[h]
        forecast_points[idx]["validated"] = True

action_list     = enforcement.build_action_list(attribution, aqi_result, osm_sites)
tomorrow_aqi    = validated_checkpoints[24]
advisory_out    = advisory.generate_advisory(ward_label or city, aqi_result["aqi"] or 150, tomorrow_aqi, language)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("PRAANA — Urban Air Quality Intelligence")
st.caption(
    f"{city} · station: {readings.get('_station_name', 'n/a')} · "
    f"data mode: {'live' if is_live and not using_fallback else 'demo / fallback'}"
)

col1, col2, col3, col4 = st.columns(4)
color     = AQI_CATEGORY_COLORS.get(aqi_result["category"], "#888888")
hex_color = color.lstrip("#")
r = int(hex_color[0:2], 16)
g = int(hex_color[2:4], 16)
b = int(hex_color[4:6], 16)

with col1:
    # FIX: AQI number was #1B3B5F (dark navy = invisible on dark bg) → #FFFFFF
    st.markdown(f"""
        <div style="padding:6px 0 10px 0;">
            <div style="color:#7FA8BC;font-size:11px;font-weight:700;
                        letter-spacing:0.10em;text-transform:uppercase;margin-bottom:6px;">
                Current AQI
            </div>
            <div style="display:flex;align-items:baseline;gap:10px;">
                <span style="color:#FFFFFF;font-size:48px;font-weight:800;line-height:1;
                             text-shadow:0 2px 14px rgba(0,0,0,0.90);">
                    {aqi_result['aqi']}
                </span>
                <span style="color:#{hex_color};font-size:16px;font-weight:700;
                             text-shadow:0 1px 8px rgba(0,0,0,0.90);">
                    {aqi_result['category'].upper()}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

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
    "Agent 1 · Source Attribution",
    "Agent 2 · Forecast",
    "Agent 3 · Enforcement",
    "Agent 4 · Multi-City",
    "Agent 5 · Citizen Advisory",
])

# ── Agent 1 ──────────────────────────────────────────────────────────────
with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Who is causing it right now")
        fig = px.pie(
            names=list(attribution["shares"].keys()),
            values=list(attribution["shares"].values()),
            color_discrete_sequence=["#4F86A0", "#E8862E", "#6FCF97", "#EB5757"],
            hole=0.38,
        )
        # FIX: textfont_color white so labels are visible on dark bg
        fig.update_traces(textinfo="percent+label", textfont_color="white", textfont_size=13)
        # FIX: transparent chart background
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#DCEAF2"),
            legend=dict(font=dict(color="#DCEAF2"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        # FIX: use_container_width replaces deprecated width='stretch'
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Confidence: **{attribution['confidence']}** · {attribution['note']}")
    with right:
        st.subheader("Live reading used")
        st.json({k: v for k, v in clean_readings.items()})
        st.subheader("CPCB sub-indices")
        st.dataframe(
            pd.DataFrame(aqi_result["sub_indices"].items(), columns=["Pollutant", "Sub-index"]),
            use_container_width=True, hide_index=True,
        )

# ── Agent 2 ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Hyperlocal AQI forecast — next 72 hours")
    df = pd.DataFrame(forecast_points)
    df["persistence_baseline"] = baseline_points[:len(df)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["hour_offset"], y=df["upper"], line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["hour_offset"], y=df["lower"], line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(232,134,46,0.18)",
                              name="Confidence band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["hour_offset"], y=df["predicted_aqi"], mode="lines",
                              line=dict(color="#E8862E", width=3), name="Predicted AQI"))
    fig.add_trace(go.Scatter(x=df["hour_offset"], y=df["persistence_baseline"], mode="lines",
                              line=dict(color="#7FA8BC", width=2, dash="dash"), name="Persistence baseline"))
    # Mark the three model-validated checkpoints (24h/48h/72h) distinctly
    # from the heuristic-interpolated curve around them.
    checkpoint_df = df[df.get("validated", False) == True] if "validated" in df.columns else df.iloc[0:0]
    if not checkpoint_df.empty:
        fig.add_trace(go.Scatter(
            x=checkpoint_df["hour_offset"], y=checkpoint_df["predicted_aqi"], mode="markers",
            marker=dict(symbol="diamond", size=11, color="#FFFFFF", line=dict(color="#E8862E", width=2)),
            name="Validated model checkpoint (24h/48h/72h)",
        ))
    # FIX: transparent background + white axis labels
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Hours from now", color="#DCEAF2",
                   gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="AQI", color="#DCEAF2",
                   gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", y=-0.22, font=dict(color="#DCEAF2"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#DCEAF2"),
        height=420, margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Diamond markers at 24h/48h/72h are real output from the independently trained, "
        "RMSE-validated per-horizon GBR models (Section 13: 80.8 / 82.5 / 83.8 vs persistence). "
        "The connecting hourly curve between checkpoints uses a lighter wind/rain/diurnal "
        "heuristic for smooth display, since the trained models only predict at those three "
        "discrete horizons. Production design uses a GNN blended with atmospheric dispersion "
        "modelling across all 72 hours."
    )

# ── Agent 3 ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Ranked enforcement actions")
    if not action_list:
        st.info("No actions ranked — source shares were all below the reporting threshold.")
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

# ── Agent 4 ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Cross-city comparison")
    if st.button("Run multi-city comparison", type="primary"):
        with st.spinner("Comparing cities..." if is_live else "Loading demo comparison..."):
            rows = multicity.compare_cities(radius_m=radius_km * 1000) if is_live else demo_data.DEMO_MULTICITY
        st.session_state["multicity_rows"] = rows

    rows = st.session_state.get("multicity_rows", demo_data.DEMO_MULTICITY if not is_live else None)
    if rows:
        df_mc   = pd.DataFrame(rows)
        plot_df = df_mc.dropna(subset=["aqi"])
        if not plot_df.empty:
            fig = px.bar(
                plot_df, x="city", y="aqi", color="category",
                color_discrete_map=AQI_CATEGORY_COLORS, text="aqi",
            )
            # FIX: transparent background + white labels
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#DCEAF2", gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(color="#DCEAF2", gridcolor="rgba(255,255,255,0.08)"),
                legend=dict(font=dict(color="#DCEAF2"), bgcolor="rgba(0,0,0,0)"),
                font=dict(color="#DCEAF2"),
                height=420, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_mc, use_container_width=True, hide_index=True)
    else:
        st.info("Click the button above to fetch a live AQI comparison across all five cities.")

# ── Agent 5 ──────────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"Citizen advisory — {language}")
    st.info(advisory_out["text"])
    groups = advisory.vulnerable_groups_flag(aqi_result["aqi"] or 0)
    if groups:
        st.warning("Flagged vulnerable groups: " + ", ".join(groups))
    else:
        st.success("No vulnerable groups flagged at this AQI level.")
