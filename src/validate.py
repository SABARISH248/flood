import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix, precision_recall_curve, auc
)
from src.utils import logger, load_config, save_run_metadata


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics given binary ground truth y_true and continuous y_pred_prob.
    """
    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_prob))
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
    except Exception:
        roc_auc = 0.0
        fpr, tpr = np.array([]), np.array([])

    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = float(auc(rec_curve, prec_curve))
    except Exception:
        pr_auc = 0.0

    precision = float(precision_score(y_true, y_pred_bin, zero_division=0))
    recall = float(recall_score(y_true, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred_bin))

    cm = confusion_matrix(y_true, y_pred_bin).tolist() if len(np.unique(y_true)) > 1 else [[0, 0], [0, 0]]

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "confusion_matrix": cm,
        "fpr": fpr.tolist() if len(fpr) < 200 else fpr[::len(fpr)//100].tolist(),
        "tpr": tpr.tolist() if len(tpr) < 200 else tpr[::len(tpr)//100].tolist()
    }


def calculate_prediction_rate(
    classified_raster: np.ndarray,
    valid_mask: np.ndarray,
    flood_points_gdf: gpd.GeoDataFrame,
    raster_transform: Any,
    class_labels: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Paper A & B Prediction Rate calculation:
    prediction_rate = (% observed flood points in class) / (% study area belonging to class)
    """
    if class_labels is None:
        class_labels = ["Very Low", "Low", "Moderate", "High", "Very High"]

    total_valid_pixels = int(np.sum(valid_mask))
    if total_valid_pixels == 0:
        return {}

    # Sample raster at flood point locations
    point_classes = []
    for geom in flood_points_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        col, row = ~raster_transform * (geom.x, geom.y)
        col_idx, row_idx = int(np.round(col)), int(np.round(row))
        if 0 <= row_idx < classified_raster.shape[0] and 0 <= col_idx < classified_raster.shape[1]:
            cls = classified_raster[row_idx, col_idx]
            if cls > 0:
                point_classes.append(cls)

    total_observed_points = len(point_classes)
    results = {}

    for c in range(1, 6):
        c_name = class_labels[c - 1] if c - 1 < len(class_labels) else f"Class {c}"
        pixel_count = int(np.sum((classified_raster == c) & valid_mask))
        area_pct = (pixel_count / total_valid_pixels) * 100.0

        point_count = point_classes.count(c)
        point_pct = (point_count / total_observed_points * 100.0) if total_observed_points > 0 else 0.0

        pred_rate = (point_pct / area_pct) if area_pct > 0 else 0.0

        results[c_name] = {
            "class_index": c,
            "pixel_count": pixel_count,
            "area_pct": round(area_pct, 2),
            "observed_points": point_count,
            "observed_pct": round(point_pct, 2),
            "prediction_rate": round(pred_rate, 4)
        }

    return results


def run_model_validation_suite(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Run metrics evaluation for multiple models on test set.
    """
    results = {}
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                p = probs[:, 1]
            else:
                p = probs.ravel()
        else:
            p = model.predict(X_test)

        results[name] = evaluate_predictions(y_test, p)

    return results
