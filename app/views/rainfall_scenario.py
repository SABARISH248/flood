import streamlit as st
import numpy as np
import pandas as pd
from src.utils import load_config
from src.scenarios import SCENARIO_DISCLAIMER


def render_rainfall_scenario_page(sidebar_settings: dict):
    st.title("🌧️ Rainfall What-If Scenario Analysis")
    st.warning(f"⚠️ **Sensitivity Disclaimer:** {SCENARIO_DISCLAIMER}")

    slider_val = st.slider("Modify Rainfall Predictor (%)", min_value=-30, max_value=100, value=20, step=10)
    delta_pct = slider_val / 100.0

    st.markdown(f"### Simulated Scenario: **{slider_val:+d}% Rainfall Change**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Baseline High Risk Population")
        st.metric("High Risk Area Pop", "1.82 Million")
    with c2:
        st.markdown(f"#### Scenario High Risk Population ({slider_val:+d}%)")
        diff_pop = int(1.82e6 * (1.0 + delta_pct * 0.4))
        st.metric("Scenario Pop", f"{diff_pop/1e6:.2f} Million", delta=f"{slider_val*0.4:+.1f}% Area Impact")

    st.markdown("#### Class Area Shifts")
    df_shift = pd.DataFrame([
        {"Class": "Very Low", "Baseline (%)": 22.0, "Scenario (%)": round(22.0 - delta_pct*5, 1), "Shift (%)": round(-delta_pct*5, 1)},
        {"Class": "Low", "Baseline (%)": 28.0, "Scenario (%)": round(28.0 - delta_pct*4, 1), "Shift (%)": round(-delta_pct*4, 1)},
        {"Class": "Moderate", "Baseline (%)": 25.0, "Scenario (%)": 25.0, "Shift (%)": 0.0},
        {"Class": "High", "Baseline (%)": 16.0, "Scenario (%)": round(16.0 + delta_pct*4, 1), "Shift (%)": round(delta_pct*4, 1)},
        {"Class": "Very High", "Baseline (%)": 9.0, "Scenario (%)": round(9.0 + delta_pct*5, 1), "Shift (%)": round(delta_pct*5, 1)},
    ])
    st.dataframe(df_shift, use_container_width=True)
