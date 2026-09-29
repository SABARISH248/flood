import streamlit as st
import pandas as pd
from src.feedback import submit_field_report, get_all_field_reports, verify_field_report


def render_field_feedback_page(sidebar_settings: dict):
    st.title("📝 Field Feedback Collection & Expert Verification")
    st.caption("Crowdsourced and ground-truth field report ingestion into SQLite + GeoPackage")

    tab1, tab2 = st.tabs(["Submit Field Report", "Admin Verification Portal"])

    with tab1:
        st.markdown("### 📥 Submit New Ground-Truth Observation")
        with st.form("feedback_form"):
            c1, c2 = st.columns(2)
            with c1:
                lat = st.number_input("Latitude", value=25.5941, format="%.4f")
                lon = st.number_input("Longitude", value=85.1376, format="%.4f")
                date_obs = st.date_input("Observation Date")
            with c2:
                depth = st.number_input("Observed Flood Depth (meters)", value=1.2, min_value=0.0, step=0.1)
                reporter = st.text_input("Reporter Name / ID", value="Field Surveyor #4")

            desc = st.text_area("Field Description & Observations", value="Road inundated, 1.2m standing water near culvert.")
            submitted = st.form_submit_button("Submit Field Report")

            if submitted:
                rec_id = submit_field_report(lat, lon, str(date_obs), depth, desc, reporter)
                st.success(f"Report ID #{rec_id} submitted successfully and logged to SQLite database!")

    with tab2:
        st.markdown("### 🛡️ Expert Verification Portal")
        df_records = get_all_field_reports()

        if not df_records.empty:
            st.dataframe(df_records, use_container_width=True)

            selected_id = st.selectbox("Select Report ID to Verify", df_records["id"].tolist())
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                notes = st.text_input("Verification Notes", value="Verified via local satellite image and ground photo")
            with col_v2:
                if st.button("Mark as Expert Verified"):
                    verify_field_report(selected_id, True, notes)
                    st.success(f"Report #{selected_id} marked as Expert Verified!")
        else:
            st.info("No field feedback records submitted yet.")
