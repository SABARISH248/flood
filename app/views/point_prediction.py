import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from src.utils import load_config
from src.data_loader import sample_raster_at_point
from src.model_registry import ModelRegistry
from src.predict import predict_probability, soft_voting, stacking_predict
from src.explain import explain_point_prediction
from src.reasons import explain_drivers, compute_background_stats
from src.geocode import geocode_place
from app.components.charts import plot_shap_waterfall
from app.components.map import create_base_map


@st.cache_data(show_spinner=False)
def _background_stats_cached(_raster_paths_tuple, _features_tuple):
    """Percentile background per feature sampled from the raster stack."""
    import rasterio
    raster_dict = {}
    raster_paths = dict(_raster_paths_tuple)
    for feat in _features_tuple:
        path = raster_paths.get(feat)
        if not path:
            continue
        try:
            with rasterio.open(path) as src:
                raster_dict[feat] = src.read(1).astype(float)
        except Exception:
            continue
    return compute_background_stats(raster_dict, list(_features_tuple))


def _place_search_box(key_prefix: str):
    """Nominatim place search with quick-select options. Returns (lat, lon, label) or None."""
    col_search, col_quick = st.columns([7, 5])
    
    with col_quick:
        quick_pick = st.selectbox(
            "⚡ Quick Pick Popular Locations",
            ["-- Select City --", "Erode, Tamil Nadu", "Chennai, Tamil Nadu", "Patna, Bihar", "Gaya, Bihar", "Muzaffarpur, Bihar", "Kochi, Kerala", "Delhi"],
            key=f"{key_prefix}_quick"
        )
        if quick_pick != "-- Select City --":
            preset_coords = {
                "Erode, Tamil Nadu": (11.3306, 77.7277),
                "Chennai, Tamil Nadu": (13.0837, 80.2702),
                "Patna, Bihar": (25.5941, 85.1376),
                "Gaya, Bihar": (24.7914, 85.0002),
                "Muzaffarpur, Bihar": (26.1209, 85.3647),
                "Kochi, Kerala": (9.9312, 76.2673),
                "Delhi": (28.6139, 77.2090)
            }
            if quick_pick in preset_coords:
                lat, lon = preset_coords[quick_pick]
                if st.session_state.get("fs_place") != quick_pick:
                    st.session_state["fs_lat"] = lat
                    st.session_state["fs_lon"] = lon
                    st.session_state["fs_place"] = quick_pick
                    st.rerun()

    with col_search:
        q = st.text_input("🔎 Search any place in India (e.g. Erode, Chennai, Patna, Gaya)",
                          key=f"{key_prefix}_q",
                          placeholder="Type city, district or landmark…")
        if st.button("Search Location", key=f"{key_prefix}_btn") and q and q.strip():
            try:
                results = geocode_place(q.strip(), limit=5)
            except ImportError:
                st.error("Place search needs `geopy`: run `pip install geopy`.")
                return None
            except (ConnectionError, ValueError) as e:
                st.error(str(e))
                return None
            if not results:
                st.warning(f"No matching place found for '{q}'. Try adding state, e.g. '{q}, Tamil Nadu'.")
                return None
            if len(results) == 1:
                r = results[0]
            else:
                labels = [r["display_name"] for r in results]
                pick = st.selectbox("Did you mean…?", labels, key=f"{key_prefix}_pick")
                r = results[labels.index(pick)]
            st.session_state["fs_lat"] = r["latitude"]
            st.session_state["fs_lon"] = r["longitude"]
            st.session_state["fs_place"] = r["display_name"]
            st.success(f"📍 {r['display_name']} → ({r['latitude']:.4f}, {r['longitude']:.4f})")
            st.rerun()
    return None


def render_point_prediction_page(sidebar_settings: dict):
    st.title("📍 Check My Location - Point Risk & Explanation")
    st.caption("Search any place or enter coordinates to compute flood susceptibility, risk drivers, and SHAP attribution")

    config = load_config()
    _place_search_box("point")

    if st.session_state.get("fs_place"):
        st.info(f"Showing results for **{st.session_state['fs_place']}**")

    c1, c2, c3 = st.columns([4, 4, 4])
    with c1:
        lat = st.number_input("Latitude (°N)", value=float(st.session_state.get("fs_lat", 25.5941)), format="%.4f")
    with c2:
        lon = st.number_input("Longitude (°E)", value=float(st.session_state.get("fs_lon", 85.1376)), format="%.4f")
    with c3:
        btn = st.button("Run Location Prediction", use_container_width=True)

    if btn or True:
        registry = ModelRegistry()
        features = registry.get_expected_features()
        raster_paths = config["rasters"]["feature_mapping"]

        point_features = sample_raster_at_point(raster_paths, features, lat, lon)
        df_point = pd.DataFrame([point_features])[features]

        base_models = registry.load_all_base_models()
        meta_learner = registry.load_meta_learner()

        base_probs = {name: round(float(predict_probability(m, df_point)[0]), 4) for name, m in base_models.items()}
        v_prob = round(float(soft_voting([np.array([p]) for p in base_probs.values()])[0]), 4)
        s_prob = round(float(stacking_predict(base_models, meta_learner, df_point)[0]), 4)

        cls_name = "Very High" if v_prob >= 0.8 else ("High" if v_prob >= 0.6 else ("Moderate" if v_prob >= 0.4 else "Low"))
        marker_color = "red" if cls_name in ["High", "Very High"] else ("orange" if cls_name == "Moderate" else "green")

        col_map, col_res = st.columns([6, 6])

        with col_map:
            st.markdown("### 🗺️ Location Map View")
            m_point = create_base_map(center=(lat, lon), zoom=12)
            popup_html = f"""
            <div style="font-family: sans-serif; font-size: 13px; width: 200px;">
                <b>📍 {st.session_state.get('fs_place', 'Selected Point')}</b><br/>
                <b>Lat, Lon:</b> {lat:.4f}, {lon:.4f}<br/>
                <b>Flood Risk Prob:</b> {v_prob * 100:.1f}%<br/>
                <b>Risk Category:</b> <span style="color: {marker_color}; font-weight: bold;">{cls_name}</span>
            </div>
            """
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{st.session_state.get('fs_place', 'Location')} - {cls_name} Risk ({v_prob * 100:.1f}%)",
                icon=folium.Icon(color=marker_color, icon="info-sign")
            ).add_to(m_point)

            st_folium(m_point, width="100%", height=420, key="point_map")

        with col_res:
            st.markdown("### 📊 Location Prediction Summary")
            st.metric("Soft Voting Flood Probability", f"{v_prob * 100:.1f}%")
            st.metric("Stacking Ensemble Probability", f"{s_prob * 100:.1f}%")

            st.markdown(f"**Susceptibility Class:** `{cls_name}`")
            st.markdown(f"**Estimated Risk Class:** `{cls_name}`")

            st.markdown("#### Per-Model Predictions")
            st.dataframe(pd.DataFrame(list(base_probs.items()), columns=["Model", "Probability"]), use_container_width=True)

        st.markdown("---")
        col_why, col_shap = st.columns([6, 6])

        with col_why:
            # ---- Why is risk high/low here? ----
            st.markdown("### ⛈️ Why is the risk at this level here?")
            st.caption("Top environmental drivers pushing risk UP or DOWN at this specific location.")
            try:
                xgb_model = base_models.get("xgboost", list(base_models.values())[0])
                explanation = explain_point_prediction(xgb_model, point_features, features)
                importances = {c["feature"]: c["shap_contribution"] for c in explanation["contributions"]}
                try:
                    bg = _background_stats_cached(tuple(sorted(raster_paths.items())), tuple(features))
                except Exception:
                    bg = None
                drivers = explain_drivers(point_features, importances, bg, top_n=5)
                if drivers:
                    for d in drivers:
                        icon = "🚨 🔺" if d["direction"] == "pushes risk up" else "🛡️ 🔻"
                        st.markdown(f"{icon} **{d['label']}**: {d['text']}")
                else:
                    st.info("Driver details unavailable for this location.")
            except Exception as e:
                st.warning(f"Driver explanation unavailable: {e}")

        with col_shap:
            st.markdown("### 🔍 SHAP Local Attribution Chart")
            xgb_model = base_models.get("xgboost", list(base_models.values())[0])
            explanation = explain_point_prediction(xgb_model, point_features, features)

            st.caption(explanation["disclaimer"])
            fig_shap = plot_shap_waterfall(explanation)
            st.plotly_chart(fig_shap, use_container_width=True)

