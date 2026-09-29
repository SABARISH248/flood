import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Any, Tuple, Optional
from src.validate import evaluate_predictions


def evaluate_historical_event(
    predicted_prob: np.ndarray,
    observed_raster_or_points: Any,
    valid_mask: np.ndarray,
    raster_transform: Any = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate predicted flood probability against an observed historical flood extent raster or point inventory.
    Calculates IoU, Hit Rate, False Alarm Rate, F1, Precision, and Recall.
    """
    if isinstance(observed_raster_or_points, np.ndarray):
        obs_raster = observed_raster_or_points
        valid_eval = valid_mask & (obs_raster >= 0)

        y_true = (obs_raster[valid_eval] > 0).astype(int)
        y_prob = predicted_prob[valid_eval]

        metrics = evaluate_predictions(y_true, y_prob, threshold=threshold)

        # Spatial IoU (Intersection over Union) for high risk binary prediction vs observed extent
        y_pred = (y_prob >= threshold).astype(int)
        intersection = np.sum((y_pred == 1) & (y_true == 1))
        union = np.sum((y_pred == 1) | (y_true == 1))
        iou = float(intersection / union) if union > 0 else 0.0

        # Hit Rate (Recall) & False Alarm Rate
        hits = intersection
        misses = np.sum((y_pred == 0) & (y_true == 1))
        false_alarms = np.sum((y_pred == 1) & (y_true == 0))
        correct_rejections = np.sum((y_pred == 0) & (y_true == 0))

        hit_rate = float(hits / (hits + misses)) if (hits + misses) > 0 else 0.0
        false_alarm_rate = float(false_alarms / (false_alarms + correct_rejections)) if (false_alarms + correct_rejections) > 0 else 0.0

        metrics.update({
            "iou": round(iou, 4),
            "hit_rate": round(hit_rate, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "total_observed_pixels": int(np.sum(y_true))
        })
        return metrics

    elif isinstance(observed_raster_or_points, gpd.GeoDataFrame) and raster_transform is not None:
        gdf = observed_raster_or_points
        sampled_probs = []

        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            col, row = ~raster_transform * (geom.x, geom.y)
            col_idx, row_idx = int(np.round(col)), int(np.round(row))
            if 0 <= row_idx < predicted_prob.shape[0] and 0 <= col_idx < predicted_prob.shape[1]:
                p = predicted_prob[row_idx, col_idx]
                if p >= 0 and valid_mask[row_idx, col_idx]:
                    sampled_probs.append(p)

        n_points = len(sampled_probs)
        mean_p = float(np.mean(sampled_probs)) if n_points > 0 else 0.0
        high_prob_count = int(np.sum(np.array(sampled_probs) >= threshold)) if n_points > 0 else 0
        hit_rate = float(high_prob_count / n_points) if n_points > 0 else 0.0

        return {
            "total_historical_points": n_points,
            "mean_predicted_probability": round(mean_p, 4),
            "points_in_high_susceptibility": high_prob_count,
            "hit_rate": round(hit_rate, 4),
            "iou": "N/A (Point Inventory)"
        }
    else:
        return {"error": "Invalid historical baseline data format."}
