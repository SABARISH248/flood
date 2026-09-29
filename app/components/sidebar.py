import streamlit as st
from typing import Dict, Any


def render_sidebar(config: Dict[str, Any]) -> Dict[str, Any]:
    """Render unified dashboard sidebar controls."""
    st.sidebar.title("🌊 FloodSense India")
    st.sidebar.caption("Spatial Flood Risk Prediction System")
    st.sidebar.markdown("---")

    st.sidebar.subheader("⚙️ Global Settings")

    selected_model = st.sidebar.selectbox(
        "Select Model / Ensemble",
        options=["soft_voting", "stacking", "xgboost", "random_forest", "lightgbm", "decision_tree", "logistic_regression"],
        index=0,
        format_func=lambda x: {
            "soft_voting": "Soft Voting Ensemble (Paper A)",
            "stacking": "Stacking Meta-Learner (Paper B)",
            "xgboost": "XGBoost Classifier",
            "random_forest": "Random Forest Classifier",
            "lightgbm": "LightGBM Classifier",
            "decision_tree": "Decision Tree Classifier",
            "logistic_regression": "Logistic Regression"
        }.get(x, x)
    )

    classification_method = st.sidebar.radio(
        "Classification Scheme",
        options=["equal_interval", "jenks"],
        index=0,
        format_func=lambda x: "Equal Interval (Paper A)" if x == "equal_interval" else "Jenks Natural Breaks (Paper B)"
    )

    map_opacity = st.sidebar.slider(
        "Raster Opacity",
        min_value=0.1,
        max_value=1.0,
        value=float(config["map"].get("opacity", 0.7)),
        step=0.05
    )

    student_mode = st.sidebar.checkbox("🎓 Explain Like I'm a Student", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Python, Rasterio, XGBoost & Streamlit")

    return {
        "model": selected_model,
        "classification": classification_method,
        "opacity": map_opacity,
        "student_mode": student_mode
    }
