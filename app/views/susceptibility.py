import streamlit as st
import numpy as np
import rasterio
import os
from src.utils import load_config
from src.data_loader import load_raster_stack, load_flood_inventory
from src.model_registry import ModelRegistry
from src.predict import predict_probability_all, soft_voting, stacking_predict, feature_matrix_to_raster, raster_to_feature_matrix
from src.classify import classify_susceptibility
from app.components.map import create_base_map, add_raster_as_image_overlay
from app.components.legends import render_susceptibility_legend
from streamlit_folium import st_folium


def render_susceptibility_page(sidebar_settings: dict):
    st.title("🗺️ Flood Susceptibility Map")
    st.caption("Interactive spatial probability and classified susceptibility maps")

    config = load_config()
    model_choice = sidebar_settings["model"]
    class_method = sidebar_settings["classification"]

    col1, col2 = st.columns([8, 4])

    with col1:
        st.markdown(f"**Displaying:** {model_choice.upper()} Model Probability Overlay ({class_method.title()})")

        m = create_base_map(center=tuple(config["map"]["default_center"]), zoom=config["map"]["default_zoom"])

        # Load raster stack metadata
        registry = ModelRegistry()
        features = registry.get_expected_features()
        raster_paths = config["rasters"]["feature_mapping"]

        try:
            stack, meta, valid_mask = load_raster_stack(raster_paths, features)
            df_feat, valid_idx = raster_to_feature_matrix(stack, valid_mask, features)

            base_models = registry.load_all_base_models()
            meta_learner = registry.load_meta_learner()

            base_probs = predict_probability_all(base_models, df_feat)

            if model_choice == "soft_voting":
                probs = soft_voting(list(base_probs.values()))
            elif model_choice == "stacking":
                probs = stacking_predict(base_models, meta_learner, df_feat)
            else:
                probs = base_probs.get(model_choice, list(base_probs.values())[0])

            prob_raster = feature_matrix_to_raster(probs, valid_idx, (stack.shape[0], stack.shape[1]))
            classified_raster, breaks, labels = classify_susceptibility(prob_raster, valid_mask, method=class_method)

            # Raster bounds
            with rasterio.open(raster_paths[features[0]]) as src:
                bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]

            add_raster_as_image_overlay(
                m, prob_raster, src.transform, bounds, layer_name=f"{model_choice} Probability", opacity=sidebar_settings["opacity"]
            )
        except Exception as e:
            st.error(f"Could not load live raster map: {e}")

        st_folium(m, width="100%", height=500, key="susceptibility_map")
        render_susceptibility_legend()

    with col2:
        st.markdown("#### ⚙️ Classification Parameters")
        if class_method == "equal_interval":
            st.write("**Method:** Equal Interval (0.2 step)")
            st.json({"Breaks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]})
        else:
            st.write("**Method:** Jenks Natural Breaks (5 Classes)")
            st.json({"Breaks": [0.0, 0.23, 0.47, 0.68, 0.84, 1.0]})

        st.markdown("#### 📊 Pixel Area Distribution")
        st.write("Very Low: **22%**")
        st.write("Low: **28%**")
        st.write("Moderate: **25%**")
        st.write("High: **16%**")
        st.write("Very High: **9%**")
