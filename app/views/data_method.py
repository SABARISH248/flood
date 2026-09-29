import streamlit as st
from src.reporting import generate_paper_ready_methods


def render_data_method_page(sidebar_settings: dict):
    st.title("📚 Data Sources, Methodology & Scientific Documentation")
    st.caption("Comprehensive technical overview and 'Explain Like I'm a Student' mode")

    student_mode = sidebar_settings.get("student_mode", False)

    if student_mode:
        st.markdown("""
        <div class="student-callout">
            <h3>🎓 Explain Like I'm a Student (Simplified Concepts)</h3>
            <ul>
                <li><b>Raster Pixel:</b> Think of the map as a giant grid of square tiles (pixels). Each tile has numbers representing elevation, rainfall, and soil type.</li>
                <li><b>Susceptibility:</b> How naturally prone an area is to flooding based on shape, height, and rain.</li>
                <li><b>Exposure:</b> People, buildings, and roads located inside a flood zone.</li>
                <li><b>Vulnerability:</b> How fragile or resilient a place is when hit by water.</li>
                <li><b>Risk:</b> Risk = Susceptibility × Exposure × Vulnerability. A flooded empty desert has low risk, but a flooded crowded city has high risk!</li>
                <li><b>Ensemble Models:</b> Instead of asking just 1 AI model, we ask 5 different models (like XGBoost and Random Forest) and average their votes for maximum accuracy!</li>
                <li><b>SHAP:</b> Tells us exactly why the AI predicted high risk for a specific town (e.g. 40% due to low elevation, 30% due to heavy rain).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📄 Paper-Ready Scientific Methodology Paragraph")
    st.code(generate_paper_ready_methods(), language="text")

    st.markdown("### 🛰️ Input Dataset Registry")
    st.table([
        {"Factor": "Elevation & Slope", "Source": "SRTM DEM 30m", "Type": "Continuous Terrain"},
        {"Factor": "Rainfall", "Source": "IMD High-Resolution Gridded Data", "Type": "Continuous Climate"},
        {"Factor": "River Discharge", "Source": "CWC Hydrological Monitoring", "Type": "Continuous Hydrological"},
        {"Factor": "Land Cover", "Source": "ISRO Bhuvan LULC", "Type": "Categorical (5 Classes)"},
        {"Factor": "Soil Type", "Source": "NBSS & LUP Soil Map", "Type": "Categorical (5 Types)"},
        {"Factor": "Historical Inventory", "Source": "Disaster Management Authority & Landsat", "Type": "Point & Polygon Vector"},
    ])
