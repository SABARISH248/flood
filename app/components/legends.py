import streamlit as st


def render_susceptibility_legend():
    """Render colorblind-safe 5-class susceptibility legend HTML."""
    html = """
    <div style="background: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #cbd5e1; margin-top: 10px;">
        <strong style="color: #1e293b; font-size: 0.9rem;">Flood Susceptibility / Risk Classes:</strong>
        <div style="display: flex; gap: 8px; margin-top: 6px; font-size: 0.8rem; font-weight: 500;">
            <div style="flex: 1; background: #2b83ba; color: white; padding: 4px; text-align: center; border-radius: 4px;">Very Low</div>
            <div style="flex: 1; background: #abdda4; color: #1e293b; padding: 4px; text-align: center; border-radius: 4px;">Low</div>
            <div style="flex: 1; background: #ffffbf; color: #1e293b; padding: 4px; text-align: center; border-radius: 4px;">Moderate</div>
            <div style="flex: 1; background: #fdae61; color: #1e293b; padding: 4px; text-align: center; border-radius: 4px;">High</div>
            <div style="flex: 1; background: #d7191c; color: white; padding: 4px; text-align: center; border-radius: 4px;">Very High</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
