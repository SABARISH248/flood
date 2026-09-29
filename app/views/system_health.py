import streamlit as st
import sys
import os
import rasterio
import sklearn
import xgboost
import lightgbm
import shap
from src.utils import load_config
from src.model_registry import ModelRegistry


def render_system_health_page(sidebar_settings: dict):
    st.title("💻 System Health & Environment Diagnostics")
    st.caption("Verifying model registry integrity, raster grid alignment, software versions, and system status")

    config = load_config()
    registry = ModelRegistry()
    meta = registry.get_metadata_summary()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 Trained Model Registry Status")
        for m in meta["available_models"]:
            status_icon = "✅" if m["exists"] else "❌"
            st.write(f"{status_icon} **{m['name']}** ({m['type']}) - Hash: `{m['file_hash']}`")

    with col2:
        st.markdown("### 📦 Installed Software Versions")
        st.write(f"Python: `{sys.version.split()[0]}`")
        st.write(f"Scikit-Learn: `{sklearn.__version__}`")
        st.write(f"XGBoost: `{xgboost.__version__}`")
        st.write(f"LightGBM: `{lightgbm.__version__}`")
        st.write(f"SHAP: `{shap.__version__}`")
        st.write(f"Rasterio: `{rasterio.__version__}`")
        st.write(f"Streamlit: `{st.__version__}`")

    st.markdown("### 🛰️ Feature Raster File Integrity")
    raster_mapping = config["rasters"]["feature_mapping"]
    raster_status = []
    for feat, path in raster_mapping.items():
        exists = os.path.exists(path)
        raster_status.append({"Feature Name": feat, "Path": path, "Status": "Found ✅" if exists else "Missing ❌"})

    st.dataframe(raster_status, use_container_width=True)
