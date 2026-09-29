import streamlit as st


def render_kpi_card(title: str, value: str, subtitle: str = ""):
    """Render a styled KPI card."""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_overview_kpis(
    total_area_km2: str = "24,500 sq km",
    high_risk_area_km2: str = "4,250 sq km",
    exposed_pop: str = "1.82 Million",
    best_auc: str = "0.942"
):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Study Area", total_area_km2, "Indian Study Region")
    with col2:
        render_kpi_card("Estimated High-Risk Area", high_risk_area_km2, "Class 4 & 5 Susceptibility")
    with col3:
        render_kpi_card("Exposed Population", exposed_pop, "High Risk Zones")
    with col4:
        render_kpi_card("Best Model ROC-AUC", best_auc, "Ensemble Voting")
