import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def plot_roc_curves(model_metrics: Dict[str, Any]):
    """Plot interactive ROC curves for evaluated models."""
    fig = go.Figure()

    for name, m in model_metrics.items():
        fpr = m.get("fpr", [])
        tpr = m.get("tpr", [])
        auc_score = m.get("roc_auc", 0.0)

        if len(fpr) > 0 and len(tpr) > 0:
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{name} (AUC = {auc_score:.3f})"
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random Chance'
    ))

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_confusion_matrix(cm_matrix: List[List[int]], model_name: str = "Voting Ensemble"):
    """Plot confusion matrix heatmap."""
    labels = ["Non-Flood", "Flood"]
    fig = px.imshow(
        cm_matrix,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        title=f"Confusion Matrix ({model_name})"
    )
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Observed Label",
        template="plotly_white"
    )
    return fig


def plot_feature_importance(importance_dict: Dict[str, float], title: str = "Feature Importance"):
    """Plot horizontal bar chart of feature importances."""
    df = pd.DataFrame(list(importance_dict.items()), columns=["Feature", "Importance"])
    df = df.sort_values(by="Importance", ascending=True)

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title=title
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_shap_waterfall(shap_explanation: Dict[str, Any]):
    """Plot SHAP contribution waterfall/bar chart for point prediction."""
    contribs = shap_explanation.get("contributions", [])
    if not contribs:
        return go.Figure()

    df = pd.DataFrame(contribs).head(10)
    df = df.iloc[::-1] # Reverse for horizontal top-down ranking

    colors = ["#ef4444" if val > 0 else "#3b82f6" for val in df["shap_contribution"]]

    fig = go.Figure(go.Bar(
        x=df["shap_contribution"],
        y=df["feature"],
        orientation='h',
        marker_color=colors,
        text=[f"{val:+.3f}" for val in df["shap_contribution"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="Local Feature Attribution (SHAP Contributions)",
        xaxis_title="Impact on Model Prediction (Log-Odds / Prob Shift)",
        yaxis_title="Feature Name",
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_priority_zones_bar(df_priority: pd.DataFrame):
    """Plot district top priority bar chart by exposed population."""
    df_top = df_priority.head(10).sort_values(by="exposed_population", ascending=True)
    fig = px.bar(
        df_top,
        x="exposed_population",
        y="name",
        orientation="h",
        color="mean_risk",
        color_continuous_scale="Reds",
        title="Top Administrative Priority Zones by Exposed Population"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Estimated Exposed Population",
        yaxis_title="Administrative Unit",
        height=450
    )
    return fig
