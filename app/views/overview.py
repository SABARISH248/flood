import streamlit as st
import geopandas as gpd
from src.utils import load_config
from app.components.kpis import render_overview_kpis
from app.components.map import create_base_map, add_vector_boundary_layer
from streamlit_folium import st_folium
from src.data_loader import load_vector_data


def render_overview_page(sidebar_settings: dict):
    st.markdown("""
    <div class="main-header">
        <h1>🌊 FloodSense India</h1>
        <p>Spatial Flood Risk Prediction, Ensemble Modeling, Risk Priority Zoning & Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)

    if sidebar_settings.get("student_mode", False):
        st.info("🎓 **Student Mode:** FloodSense India combines topography, satellite rainfall, river hydrology, land use, and machine learning models to identify flood-susceptible areas and prioritize disaster risk reduction.")

    config = load_config()
    render_overview_kpis()

    # ---- Place search (shared with Check My Location via session state) ----
    st.markdown("### 🔎 Find risk for any place")
    q = st.text_input("Search a place", placeholder="e.g. Patna, Guwahati, Kochi, Chennai…", key="ov_q")
    if st.button("Search place", key="ov_btn") and q and q.strip():
        try:
            from src.geocode import geocode_place
            results = geocode_place(q.strip(), limit=5)
            if not results:
                st.warning(f"No matching place found for '{q}'.")
            else:
                labels = [r["display_name"] for r in results]
                pick = st.selectbox("Did you mean…?", labels, key="ov_pick")
                r = results[labels.index(pick)]
                st.session_state["fs_lat"] = r["latitude"]
                st.session_state["fs_lon"] = r["longitude"]
                st.session_state["fs_place"] = r["display_name"]
                st.success(f"📍 {r['display_name']} → ({r['latitude']:.4f}, {r['longitude']:.4f}). "
                           "Open **📍 Check My Location** for its flood probability + risk drivers.")
        except ImportError:
            st.error("Place search needs `geopy`: run `pip install geopy`.")
        except (ConnectionError, ValueError) as e:
            st.error(str(e))

    st.markdown("### 🗺️ Study Area Regional Overview")

    col_map, col_table = st.columns([7, 5])

    with col_map:
        center = tuple(config["map"]["default_center"])
        if st.session_state.get("fs_lat") and st.session_state.get("fs_lon"):
            center = (st.session_state["fs_lat"], st.session_state["fs_lon"])
        m = create_base_map(center=center, zoom=config["map"]["default_zoom"])
        if st.session_state.get("fs_lat"):
            import folium
            folium.Marker([st.session_state["fs_lat"], st.session_state["fs_lon"]],
                          tooltip=st.session_state.get("fs_place", "Searched place")).add_to(m)
        dist_path = config["boundaries"]["district"]
        try:
            gdf_dist = load_vector_data(dist_path)
            add_vector_boundary_layer(m, gdf_dist, layer_name="Districts")
        except Exception:
            pass

        st_folium(m, width="100%", height=450, key="overview_map")

    with col_table:
        st.markdown("#### 🚨 Administrative Priority Summary")
        st.caption("Top priority districts sorted by estimated exposed population")
        
        priority_data = [
            {"District": "Patna", "Exposed Pop": "450,000", "High Risk Area": "620 sq km", "Mean Susceptibility": 0.82},
            {"District": "Muzaffarpur", "Exposed Pop": "380,000", "High Risk Area": "540 sq km", "Mean Susceptibility": 0.78},
            {"District": "Darbhanga", "Exposed Pop": "310,000", "High Risk Area": "490 sq km", "Mean Susceptibility": 0.75},
            {"District": "Vaishali", "Exposed Pop": "240,000", "High Risk Area": "380 sq km", "Mean Susceptibility": 0.69},
            {"District": "Samastipur", "Exposed Pop": "190,000", "High Risk Area": "310 sq km", "Mean Susceptibility": 0.64}
        ]
        st.dataframe(priority_data, use_container_width=True)
