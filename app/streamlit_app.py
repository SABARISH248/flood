import sys
import os

# Ensure project root is in sys.path so 'src' and 'app' packages resolve properly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Change working directory to project root so config.yaml and all relative paths resolve
os.chdir(ROOT_DIR)

import streamlit as st

# Set page layout configuration
st.set_page_config(
    page_title="FloodSense India",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from src.utils import load_config
from app.components.sidebar import render_sidebar

# Import all page renderers from views/ (renamed from pages/ to prevent Streamlit auto-discovery)
from app.views.overview import render_overview_page
from app.views.susceptibility import render_susceptibility_page
from app.views.risk_priority import render_risk_priority_page
from app.views.point_prediction import render_point_prediction_page
from app.views.rainfall_scenario import render_rainfall_scenario_page
from app.views.model_validation import render_model_validation_page
from app.views.uncertainty import render_uncertainty_page
from app.views.historical_events import render_historical_events_page
from app.views.field_feedback import render_field_feedback_page
from app.views.data_method import render_data_method_page
from app.views.system_health import render_system_health_page


def main():
    config = load_config()
    sidebar_settings = render_sidebar(config)

    # Page Navigation menu
    page_menu = {
        "📊 Overview": render_overview_page,
        "🗺️ Susceptibility Map": render_susceptibility_page,
        "🎯 Risk & Priority Zones": render_risk_priority_page,
        "📍 Check My Location": render_point_prediction_page,
        "🌧️ Rainfall Scenario": render_rainfall_scenario_page,
        "📈 Model Validation": render_model_validation_page,
        "⚖️ Model Uncertainty": render_uncertainty_page,
        "📜 Historical Events": render_historical_events_page,
        "📝 Field Feedback": render_field_feedback_page,
        "📚 Data & Method": render_data_method_page,
        "💻 System Health": render_system_health_page
    }

    selected_page_name = st.sidebar.radio("🧭 Navigation", list(page_menu.keys()), index=0)

    # Render selected page with full error handling
    try:
        render_func = page_menu[selected_page_name]
        render_func(sidebar_settings)
    except Exception as e:
        st.error(f"⚠️ Page error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
