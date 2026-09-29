import streamlit as st
import pandas as pd
from src.utils import load_config
from app.components.charts import plot_priority_zones_bar
from src.reporting import generate_pdf_report


def render_risk_priority_page(sidebar_settings: dict):
    st.title("🎯 Risk & Priority Zone Analysis")
    st.caption("Risk = Susceptibility × Exposure × Vulnerability")

    config = load_config()

    col1, col2 = st.columns([7, 5])

    with col1:
        st.markdown("### Risk Layer Formula")
        r_method = st.radio("Risk Formula Method", ["Weighted Overlay (Ws*S + We*E + Wv*V)", "Multiplicative (S * E * V)"], index=0)
        
        st.markdown("#### Priority Administrative Units Table")
        df_priority = pd.DataFrame([
            {"name": "Patna", "total_area_km2": 3202.0, "high_risk_area_km2": 620.0, "pct_high_risk": 19.36, "exposed_population": 450000, "exposed_buildings": 42000, "exposed_road_km": 145.2, "hospitals_count": 18, "schools_count": 85, "mean_risk": 0.72},
            {"name": "Muzaffarpur", "total_area_km2": 3172.0, "high_risk_area_km2": 540.0, "pct_high_risk": 17.02, "exposed_population": 380000, "exposed_buildings": 34000, "exposed_road_km": 118.5, "hospitals_count": 14, "schools_count": 72, "mean_risk": 0.68},
            {"name": "Darbhanga", "total_area_km2": 2279.0, "high_risk_area_km2": 490.0, "pct_high_risk": 21.50, "exposed_population": 310000, "exposed_buildings": 29000, "exposed_road_km": 94.0, "hospitals_count": 11, "schools_count": 64, "mean_risk": 0.65},
            {"name": "Vaishali", "total_area_km2": 2036.0, "high_risk_area_km2": 380.0, "pct_high_risk": 18.66, "exposed_population": 240000, "exposed_buildings": 22000, "exposed_road_km": 82.4, "hospitals_count": 9, "schools_count": 51, "mean_risk": 0.61},
            {"name": "Samastipur", "total_area_km2": 2904.0, "high_risk_area_km2": 310.0, "pct_high_risk": 10.67, "exposed_population": 190000, "exposed_buildings": 18000, "exposed_road_km": 68.1, "hospitals_count": 8, "schools_count": 46, "mean_risk": 0.58}
        ])

        st.dataframe(df_priority, use_container_width=True)

        st.markdown("#### 📥 Export Center")
        col_csv, col_pdf = st.columns(2)
        with col_csv:
            csv_str = df_priority.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Table", data=csv_str, file_name="priority_zones.csv", mime="text/csv")
        with col_pdf:
            if st.button("Generate PDF Report"):
                pdf_path = generate_pdf_report({"run_id": "RUN-2026-09-28-001", "metrics": {}})
                st.success(f"Generated PDF: {pdf_path}")

    with col2:
        fig_bar = plot_priority_zones_bar(df_priority)
        st.plotly_chart(fig_bar, use_container_width=True)
