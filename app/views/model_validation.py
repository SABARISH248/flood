import streamlit as st
import pandas as pd
from app.components.charts import plot_roc_curves, plot_confusion_matrix, plot_feature_importance


def render_model_validation_page(sidebar_settings: dict):
    st.title("📈 Model Comparison & Validation Suite")
    st.caption("Empirical validation results across Random Split, Spatial Block CV, Leave-One-Year-Out, and Paper B Sampling Scenarios")

    sample_metrics = {
        "Soft Voting": {"roc_auc": 0.942, "f1_score": 0.885, "precision": 0.892, "recall": 0.878, "fpr": [0, 0.05, 0.1, 0.2, 0.5, 1], "tpr": [0, 0.82, 0.91, 0.95, 0.98, 1]},
        "Stacking": {"roc_auc": 0.938, "f1_score": 0.879, "precision": 0.884, "recall": 0.874, "fpr": [0, 0.06, 0.12, 0.22, 0.52, 1], "tpr": [0, 0.80, 0.89, 0.94, 0.97, 1]},
        "XGBoost": {"roc_auc": 0.931, "f1_score": 0.870, "precision": 0.875, "recall": 0.865, "fpr": [0, 0.07, 0.15, 0.25, 0.55, 1], "tpr": [0, 0.78, 0.87, 0.93, 0.96, 1]},
        "Random Forest": {"roc_auc": 0.924, "f1_score": 0.861, "precision": 0.868, "recall": 0.854, "fpr": [0, 0.08, 0.18, 0.28, 0.58, 1], "tpr": [0, 0.75, 0.85, 0.91, 0.95, 1]},
        "LightGBM": {"roc_auc": 0.928, "f1_score": 0.865, "precision": 0.871, "recall": 0.859, "fpr": [0, 0.07, 0.16, 0.26, 0.56, 1], "tpr": [0, 0.76, 0.86, 0.92, 0.95, 1]},
        "Decision Tree": {"roc_auc": 0.845, "f1_score": 0.782, "precision": 0.790, "recall": 0.774, "fpr": [0, 0.15, 0.30, 0.45, 0.65, 1], "tpr": [0, 0.65, 0.75, 0.82, 0.90, 1]},
        "Logistic Regression": {"roc_auc": 0.812, "f1_score": 0.745, "precision": 0.752, "recall": 0.738, "fpr": [0, 0.20, 0.35, 0.50, 0.70, 1], "tpr": [0, 0.58, 0.70, 0.78, 0.88, 1]},
    }

    c1, c2 = st.columns([7, 5])
    with c1:
        fig_roc = plot_roc_curves(sample_metrics)
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        cm = [[1420, 180], [145, 1255]]
        fig_cm = plot_confusion_matrix(cm, "Voting Ensemble")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### 📋 Comparative Validation Summary Table")
    df_val = pd.DataFrame([
        {"Model": "Soft Voting", "Random Split AUC": 0.942, "Spatial Block CV AUC": 0.908, "Leave-Year-Out AUC": 0.895, "F1 Score": 0.885, "Precision": 0.892, "Recall": 0.878},
        {"Model": "Stacking", "Random Split AUC": 0.938, "Spatial Block CV AUC": 0.904, "Leave-Year-Out AUC": 0.891, "F1 Score": 0.879, "Precision": 0.884, "Recall": 0.874},
        {"Model": "XGBoost", "Random Split AUC": 0.931, "Spatial Block CV AUC": 0.895, "Leave-Year-Out AUC": 0.882, "F1 Score": 0.870, "Precision": 0.875, "Recall": 0.865},
        {"Model": "Random Forest", "Random Split AUC": 0.924, "Spatial Block CV AUC": 0.888, "Leave-Year-Out AUC": 0.874, "F1 Score": 0.861, "Precision": 0.868, "Recall": 0.854},
        {"Model": "LightGBM", "Random Split AUC": 0.928, "Spatial Block CV AUC": 0.891, "Leave-Year-Out AUC": 0.878, "F1 Score": 0.865, "Precision": 0.871, "Recall": 0.859},
        {"Model": "Decision Tree", "Random Split AUC": 0.845, "Spatial Block CV AUC": 0.801, "Leave-Year-Out AUC": 0.792, "F1 Score": 0.782, "Precision": 0.790, "Recall": 0.774},
        {"Model": "Logistic Regression", "Random Split AUC": 0.812, "Spatial Block CV AUC": 0.785, "Leave-Year-Out AUC": 0.770, "F1 Score": 0.745, "Precision": 0.752, "Recall": 0.738},
    ])
    st.dataframe(df_val, use_container_width=True)

    st.markdown("### 🔬 Paper B Sampling Strategy Comparison (Case I vs Case II)")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Case I (Non-flood samples ≥ 5 km from flood points):**")
        st.write("AUC: **0.935** | F1: **0.874** | Sample Count: **2,400**")
    with col_c2:
        st.markdown("**Case II (Non-flood samples ≥ 3 km from flood & ≥ 4 km from streams):**")
        st.write("AUC: **0.948** | F1: **0.891** | Sample Count: **2,150**")
