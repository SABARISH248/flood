import os
import json
import numpy as np
import jenkspy
from typing import Dict, Any, List, Tuple, Optional
from src.utils import logger, load_config


DEFAULT_EQUAL_BREAKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]


def equal_interval_classify(
    prob_raster: np.ndarray,
    valid_mask: np.ndarray,
    breaks: Optional[List[float]] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify probability raster into 5 susceptibility classes using Equal Interval.
    Default breaks: 0.0 - 0.2 - 0.4 - 0.6 - 0.8 - 1.0 -> Classes 1 to 5.
    """
    if breaks is None:
        breaks = DEFAULT_EQUAL_BREAKS

    classified = np.zeros_like(prob_raster, dtype=np.int32)
    valid_probs = prob_raster[valid_mask]

    # Assign class 1 to 5 based on intervals
    for i in range(len(breaks) - 1):
        low, high = breaks[i], breaks[i + 1]
        if i == len(breaks) - 2:
            mask = (prob_raster >= low) & (prob_raster <= high) & valid_mask
        else:
            mask = (prob_raster >= low) & (prob_raster < high) & valid_mask
        classified[mask] = i + 1

    classified[~valid_mask] = 0
    return classified, breaks


def jenks_classify(
    prob_raster: np.ndarray,
    valid_mask: np.ndarray,
    n_classes: int = 5,
    sample_size: int = 10000
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify probability raster using Jenks Natural Breaks optimization.
    Uses sub-sampling if total valid pixels > sample_size to speed up calculation.
    """
    valid_vals = prob_raster[valid_mask].astype(float)
    if len(valid_vals) == 0:
        breaks = [0.0] * (n_classes + 1)
        return np.zeros_like(prob_raster, dtype=np.int32), breaks

    if len(valid_vals) > sample_size:
        np.random.seed(42)
        sample_vals = np.random.choice(valid_vals, size=sample_size, replace=False)
    else:
        sample_vals = valid_vals

    try:
        breaks = jenkspy.jenks_breaks(sample_vals, n_classes=n_classes)
    except Exception as e:
        logger.warning(f"Jenks calculation fallback to quantiles due to: {e}")
        quantiles = np.linspace(0, 100, n_classes + 1)
        breaks = list(np.percentile(valid_vals, quantiles))

    # Ensure monotonic bounds 0.0 to 1.0
    breaks[0] = min(0.0, float(breaks[0]))
    breaks[-1] = max(1.0, float(breaks[-1]))

    classified = np.zeros_like(prob_raster, dtype=np.int32)
    for i in range(len(breaks) - 1):
        low, high = breaks[i], breaks[i + 1]
        if i == len(breaks) - 2:
            mask = (prob_raster >= low) & (prob_raster <= high) & valid_mask
        else:
            mask = (prob_raster >= low) & (prob_raster < high) & valid_mask
        classified[mask] = i + 1

    classified[~valid_mask] = 0
    return classified, breaks


def classify_susceptibility(
    prob_raster: np.ndarray,
    valid_mask: np.ndarray,
    method: str = "equal_interval",
    breaks: Optional[List[float]] = None,
    n_classes: int = 5
) -> Tuple[np.ndarray, List[float], List[str]]:
    """
    Unified entry point for classifying probability raster.
    """
    labels = DEFAULT_LABELS
    if method == "jenks":
        classified, calc_breaks = jenks_classify(prob_raster, valid_mask, n_classes=n_classes)
    else:
        classified, calc_breaks = equal_interval_classify(prob_raster, valid_mask, breaks=breaks)

    return classified, calc_breaks, labels


def save_breaks_to_json(breaks: List[float], labels: List[str], output_path: str) -> str:
    """Save calculated breaks metadata to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "breaks": [round(float(b), 4) for b in breaks],
        "labels": labels
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved classification breaks to {output_path}")
    return output_path
