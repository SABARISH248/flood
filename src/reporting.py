import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from src.utils import logger, load_config


def generate_results_table(metrics_dict: Dict[str, Any], output_dir: str = "outputs/reports") -> Tuple[str, str]:
    """
    Generate paper-ready results table CSV and JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for model_name, m in metrics_dict.items():
        rows.append({
            "Model": model_name,
            "AUC": m.get("roc_auc", "N/A"),
            "F1 Score": m.get("f1_score", "N/A"),
            "Precision": m.get("precision", "N/A"),
            "Recall": m.get("recall", "N/A"),
            "Spatial CV AUC": m.get("spatial_cv_auc_mean", "N/A"),
            "Leave-Year-Out AUC": m.get("year_out_auc_mean", "N/A")
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "results_table.csv")
    json_path = os.path.join(output_dir, "results.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Saved results table to {csv_path} and {json_path}")
    return csv_path, json_path


def generate_paper_ready_methods(config_path: str = "config.yaml") -> str:
    """
    Generate a paper-ready methodology paragraph based strictly on actual configured features and models.
    """
    config = load_config(config_path)
    features = config["features"]["exact_order"]
    models = config["models"]["base_models"]
    crs = config["CRS"]["analysis"]

    paragraph = (
        f"Flood susceptibility modeling was conducted using an ensemble of machine learning algorithms "
        f"including {', '.join(models)}. The feature space comprised {len(features)} spatial conditioning factors "
        f"({', '.join(features)}), standardized to a common grid in {crs} projection. "
        f"Ensemble predictions were generated using both pixel-wise probability soft voting and a meta-learner stacking strategy. "
        f"Classification into five susceptibility levels was executed using Equal Interval and Jenks Natural Breaks optimization. "
        f"Spatial block cross-validation and feature importance analysis via SHAP and permutation importance were evaluated "
        f"to prevent spatial data leakage and assess feature attributions."
    )
    return paragraph


def generate_pdf_report(
    summary_data: Dict[str, Any],
    output_pdf_path: str = "outputs/reports/FloodSense_Report.pdf"
) -> str:
    """
    Generate a professional PDF summary report using ReportLab.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = styles['Normal']

    story = []

    # Title Banner
    story.append(Paragraph("🌊 FloodSense India - Flood Risk Prediction Report", title_style))
    story.append(Spacer(1, 10))

    # Project Context
    run_id = summary_data.get("run_id", "N/A")
    date_str = summary_data.get("timestamp", "2026-09-28")
    story.append(Paragraph(f"<b>Run Identifier:</b> {run_id} | <b>Generated:</b> {date_str}", body_style))
    story.append(Spacer(1, 12))

    # Methodology Section
    story.append(Paragraph("Executive Methodology Summary", h2_style))
    methods_text = generate_paper_ready_methods()
    story.append(Paragraph(methods_text, body_style))
    story.append(Spacer(1, 12))

    # Evaluation Metrics Table
    metrics = summary_data.get("metrics", {})
    if metrics:
        story.append(Paragraph("Model Performance Metrics", h2_style))
        table_data = [["Model", "ROC-AUC", "F1 Score", "Precision", "Recall"]]
        for name, m in metrics.items():
            table_data.append([
                name,
                str(m.get("roc_auc", "N/A")),
                str(m.get("f1_score", "N/A")),
                str(m.get("precision", "N/A")),
                str(m.get("recall", "N/A"))
            ])

        t = Table(table_data, colWidths=[120, 80, 80, 80, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F7FAFC')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

    # Limitations & Disclaimers
    story.append(Paragraph("Scientific Limitations & Disclaimers", h2_style))
    disclaimer_text = (
        "1. Susceptibility models represent long-term spatial likelihood based on static terrain and historical factors, "
        "not real-time hydrological flood forecasts.<br/>"
        "2. SHAP feature contributions quantify statistical associations and must not be interpreted as direct causal drivers.<br/>"
        "3. Field verification by local water resource authorities is recommended prior to emergency policy deployment."
    )
    story.append(Paragraph(disclaimer_text, body_style))

    doc.build(story)
    logger.info(f"Generated PDF report at {output_pdf_path}")
    return output_pdf_path
