import streamlit as st
import numpy as np
import pandas as pd
from src.utils import load_config
from src.data_loader import sample_raster_at_point
from src.model_registry import ModelRegistry
from src.predict import predict_probability, soft_voting, stacking_predict
from src.explain import explain_point_prediction
from app.components.charts import plot_shap_waterfall


def render_point_prediction_page(sidebar_settings: dict):
    st.title("📍 Check My Location - Point Prediction & Explanation")
    st.caption("Click map or enter coordinates to compute location flood susceptibility and SHAP attribution")

    config = load_config()

    c1, c2, c3 = st.columns([4, 4, 4])
    with c1:
        lat = st.number_input("Latitude (°N)", value=25.5941, format="%.4f")
    with c2:
        lon = st.number_input("Longitude (°E)", value=85.1376, format="%.4f")
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

        col_res, col_shap = st.columns([5, 7])

        with col_res:
            st.markdown("### 📊 Location Prediction Summary")
            st.metric("Soft Voting Flood Probability", f"{v_prob * 100:.1f}%")
            st.metric("Stacking Ensemble Probability", f"{s_prob * 100:.1f}%")

            cls_name = "Very High" if v_prob >= 0.8 else ("High" if v_prob >= 0.6 else ("Moderate" if v_prob >= 0.4 else "Low"))
            st.markdown(f"**Susceptibility Class:** `{cls_name}`")
            st.markdown(f"**Estimated Risk Class:** `{cls_name}`")

            st.markdown("#### Per-Model Predictions")
            st.dataframe(pd.DataFrame(list(base_probs.items()), columns=["Model", "Probability"]), use_container_width=True)

        with col_shap:
            st.markdown("### 🔍 SHAP Local Attribution")
            xgb_model = base_models.get("xgboost", list(base_models.values())[0])
            explanation = explain_point_prediction(xgb_model, point_features, features)

            st.caption(explanation["disclaimer"])
            fig_shap = plot_shap_waterfall(explanation)
            st.plotly_chart(fig_shap, use_container_width=True)
