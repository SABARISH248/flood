"""
Tamil Nadu District-Level Flood Risk Prediction Page
──────────────────────────────────────────────────────
Select any of the 38 Tamil Nadu districts. The page will:
  1. Locate the district on the map (verified coordinates).
  2. Display its district boundary bounding box.
  3. Fetch environmental features (rainfall, elevation, land cover, etc.)
  4. Run the trained ML ensemble through the FastAPI backend.
  5. Display the actual model prediction — not a hardcoded number.
"""
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import requests
import json

API_ROOT = "http://127.0.0.1:8000/api/v1"

RISK_COLORS = {
    "Very High": "#d32f2f",
    "High": "#f57c00",
    "Moderate": "#fbc02d",
    "Low": "#388e3c",
}

RISK_EMOJIS = {
    "Very High": "🔴",
    "High": "🟠",
    "Moderate": "🟡",
    "Low": "🟢",
}


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_districts():
    try:
        r = requests.get(f"{API_ROOT}/districts", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _predict_district(district_id: int):
    try:
        r = requests.post(
            f"{API_ROOT}/predict/district",
            json={"district_id": district_id},
            timeout=10,
        )
        r.raise_for_status()
        return r.json(), None
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


def _build_map(centroid: dict, bbox: list, risk_label: str, district_name: str, prob_pct: float):
    lat, lon = centroid["lat"], centroid["lon"]
    color = RISK_COLORS.get(risk_label, "#757575")
    m = folium.Map(
        location=[lat, lon],
        zoom_start=9,
        tiles="CartoDB positron",
    )
    # District bounding box rectangle
    south, west, north, east = bbox[0], bbox[1], bbox[2], bbox[3]
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color=color,
        weight=3,
        fill=True,
        fill_color=color,
        fill_opacity=0.15,
        tooltip=f"{district_name} – {risk_label} Risk",
    ).add_to(m)
    # Centroid marker
    popup_html = f"""
    <div style="font-family:sans-serif;font-size:13px;width:220px">
        <b>📍 {district_name}</b><br/>
        <b>Flood Risk:</b>
        <span style="color:{color};font-weight:bold">{risk_label}</span><br/>
        <b>Probability:</b> {prob_pct:.1f}%<br/>
        <b>Centroid:</b> {lat:.4f}°N, {lon:.4f}°E
    </div>
    """
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"{district_name}: {risk_label} ({prob_pct:.1f}%)",
        icon=folium.Icon(color="red" if risk_label in ("Very High", "High") else "orange" if risk_label == "Moderate" else "green", icon="info-sign"),
    ).add_to(m)
    return m


def render_tn_district_page(sidebar_settings: dict):
    st.title("🗺️ Tamil Nadu District Flood Risk Prediction")
    st.caption(
        "Select a Tamil Nadu district → the system fetches its environmental data → "
        "runs the trained ML ensemble → shows the actual flood-risk prediction."
    )

    # ── Fetch district list ────────────────────────────────────────────────
    districts_data = _fetch_districts()
    if isinstance(districts_data, dict) and "error" in districts_data:
        st.error(f"⚠️ Cannot reach backend API: {districts_data['error']}\n\nMake sure `uvicorn api.main:app` is running.")
        return

    districts = districts_data  # list of {id, name, centroid, area_km2}
    district_names = [d["name"] for d in districts]

    # ── District selector ──────────────────────────────────────────────────
    col_sel, col_btn = st.columns([7, 3])
    with col_sel:
        selected_name = st.selectbox(
            "🔍 Choose a Tamil Nadu District",
            ["— Select district —"] + district_names,
            key="tn_district_select",
        )
    with col_btn:
        st.markdown("<br/>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict Flood Risk", use_container_width=True, type="primary")

    if selected_name == "— Select district —":
        # Empty state
        st.info("👆 Select a district above to see its flood risk prediction.")
        # Show Tamil Nadu overview map
        m0 = folium.Map(location=[10.8, 78.7], zoom_start=7, tiles="CartoDB positron")
        for d in districts:
            c = d["centroid"]
            folium.CircleMarker(
                location=[c["lat"], c["lon"]],
                radius=5,
                color="#1565c0",
                fill=True,
                fill_opacity=0.6,
                tooltip=d["name"],
            ).add_to(m0)
        st_folium(m0, width="100%", height=480, key="tn_overview_map")
        return

    # ── Prediction ────────────────────────────────────────────────────────
    selected_district = next(d for d in districts if d["name"] == selected_name)
    district_id = selected_district["id"]

    # Auto-predict on selection change OR button press
    pred_key = f"tn_pred_{district_id}"
    if predict_btn or pred_key not in st.session_state:
        with st.spinner(f"Running ML prediction for **{selected_name}**…"):
            payload, err = _predict_district(district_id)
        if err:
            st.error(f"❌ Prediction failed: {err}")
            return
        st.session_state[pred_key] = payload

    data = st.session_state.get(pred_key)
    if not data:
        return

    pred = data["prediction"]
    prob_pct = pred["probability_percent"]
    risk_label = pred["risk_label"]
    risk_color = RISK_COLORS.get(risk_label, "#757575")
    risk_emoji = RISK_EMOJIS.get(risk_label, "⚪")

    # ── Layout: Map + Results ──────────────────────────────────────────────
    col_map, col_res = st.columns([6, 6])

    with col_map:
        st.markdown("### 🗺️ District Map")
        m = _build_map(
            centroid=data["centroid"],
            bbox=data["bbox"],
            risk_label=risk_label,
            district_name=data["district_name"],
            prob_pct=prob_pct,
        )
        st_folium(m, width="100%", height=430, key=f"tn_map_{district_id}")

    with col_res:
        st.markdown(f"### {risk_emoji} Flood Risk: **{selected_name}**")

        # Big probability metric
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{risk_color}22,{risk_color}44);
                        border-left:5px solid {risk_color};
                        border-radius:10px;padding:18px 22px;margin:8px 0">
                <div style="font-size:13px;color:#555">Flood Susceptibility Probability</div>
                <div style="font-size:48px;font-weight:900;color:{risk_color}">{prob_pct:.1f}%</div>
                <div style="font-size:20px;font-weight:700;color:{risk_color}">{risk_emoji} {risk_label} Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**District:** {data['district_name']}")
        st.markdown(f"**Centroid:** {data['centroid']['lat']:.4f}°N, {data['centroid']['lon']:.4f}°E")
        st.markdown(f"**Area:** {data['area_km2']} km²")
        st.markdown(f"**Stacking Ensemble:** {pred['stacking_probability']*100:.1f}%")
        st.markdown(f"**Soft Voting:** {pred['soft_voting_probability']*100:.1f}%")

        # Data note
        if data.get("data_note"):
            st.info(f"📌 {data['data_note']}")

    # ── Environmental Features ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌍 Environmental Input Features")
    st.caption(
        "These are district-level statistics from published datasets (IMD, SRTM, NLCMS, Census, NFDB) "
        "— not randomly generated."
    )

    env = data["environmental_features"]
    env_cols = [
        ("🌧️ Rainfall (mm)", env.get("Rainfall (mm)")),
        ("🌡️ Temperature (°C)", env.get("Temperature (°C)")),
        ("💧 Humidity (%)", env.get("Humidity (%)")),
        ("🏔️ Elevation (m)", env.get("Elevation (m)")),
        ("👥 Population Density", env.get("Population Density")),
        ("🌊 River Discharge (m³/s)", env.get("River Discharge (m³/s)")),
        ("📏 Water Level (m)", env.get("Water Level (m)")),
        ("🌾 Agricultural Cover", f"{env.get('Land Cover_Agricultural',0)*100:.0f}%"),
        ("🌳 Forest Cover", f"{env.get('Land Cover_Forest',0)*100:.0f}%"),
        ("🏙️ Urban Cover", f"{env.get('Land Cover_Urban',0)*100:.0f}%"),
        ("📜 Historical Floods", "Yes ✅" if env.get("Historical Floods") else "No ❌"),
    ]

    col1, col2, col3 = st.columns(3)
    for i, (label, value) in enumerate(env_cols):
        with [col1, col2, col3][i % 3]:
            st.metric(label, value)

    # ── Per-Model Probabilities ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Per-Model Flood Probability")
    per_model = {k: v for k, v in data.get("per_model_probabilities", {}).items() if v is not None}
    if per_model:
        df_models = pd.DataFrame(
            [(name.replace("_", " ").title(), f"{p*100:.1f}%", _risk_label(p)) for name, p in per_model.items()],
            columns=["Model", "Probability", "Risk Level"],
        )
        st.dataframe(df_models, use_container_width=True, hide_index=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️ **Prototype notice:** The ML model was trained on a synthetic normalised dataset. "
        "Predictions are flood-occurrence susceptibility estimates derived from district-level environmental statistics. "
        "They are **not** certified hazard probabilities. Independent validation with real historical flood records "
        "is required before operational use."
    )
    st.caption(f"🕐 Prediction timestamp: {data.get('prediction_timestamp', 'N/A')}")
    st.caption(f"📚 Data sources: {data.get('data_sources', 'N/A')}")


def _risk_label(p: float) -> str:
    if p >= 0.75: return "Very High"
    if p >= 0.55: return "High"
    if p >= 0.35: return "Moderate"
    return "Low"
