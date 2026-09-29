import os
import numpy as np
from typing import Dict, List, Tuple, Any
from src.utils import logger
from src.predict import write_probability_geotiff


def calculate_pixel_uncertainty(
    prob_rasters_dict: Dict[str, np.ndarray],
    valid_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate pixel-wise uncertainty and model agreement metrics across multiple model predictions.
    Returns: (mean_prob, std_dev, min_prob, max_prob, agreement_pct)
    """
    raster_list = [arr for key, arr in prob_rasters_dict.items() if key not in ["soft_voting", "stacking"]]
    if not raster_list:
        raster_list = list(prob_rasters_dict.values())

    stacked = np.stack(raster_list, axis=0) # Shape: (N_models, height, width)

    mean_prob = np.mean(stacked, axis=0).astype(np.float32)
    std_dev = np.std(stacked, axis=0).astype(np.float32)
    min_prob = np.min(stacked, axis=0).astype(np.float32)
    max_prob = np.max(stacked, axis=0).astype(np.float32)

    # Class agreement: Convert each model prob to class (1-5 via equal interval 0.2 step)
    # class 1: [0, 0.2), 2: [0.2, 0.4), 3: [0.4, 0.6), 4: [0.6, 0.8), 5: [0.8, 1.0]
    classes = np.clip(np.floor(stacked / 0.2).astype(int) + 1, 1, 5)

    # Calculate majority class count per pixel
    # Count frequency of each class per pixel
    n_models = stacked.shape[0]
    agreement = np.zeros(mean_prob.shape, dtype=np.float32)

    h, w = mean_prob.shape
    for i in range(h):
        for j in range(w):
            if valid_mask[i, j]:
                pixel_classes = classes[:, i, j]
                bincount = np.bincount(pixel_classes, minlength=6)
                max_votes = np.max(bincount)
                agreement[i, j] = (max_votes / n_models) * 100.0

    mean_prob[~valid_mask] = -9999.0
    std_dev[~valid_mask] = -9999.0
    min_prob[~valid_mask] = -9999.0
    max_prob[~valid_mask] = -9999.0
    agreement[~valid_mask] = -9999.0

    return mean_prob, std_dev, min_prob, max_prob, agreement


def generate_uncertainty_outputs(
    prob_rasters_dict: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    target_meta: Dict[str, Any],
    output_dir: str = "outputs/uncertainty"
) -> Dict[str, str]:
    """
    Generate and save GeoTIFF files for std_dev uncertainty and class agreement percentage.
    """
    os.makedirs(output_dir, exist_ok=True)
    mean_p, std_p, min_p, max_p, agreement_pct = calculate_pixel_uncertainty(prob_rasters_dict, valid_mask)

    std_path = os.path.join(output_dir, "uncertainty_std.tif")
    agree_path = os.path.join(output_dir, "model_agreement.tif")

    write_probability_geotiff(std_p, target_meta, std_path)
    write_probability_geotiff(agreement_pct, target_meta, agree_path)

    return {
        "uncertainty_std_path": std_path,
        "agreement_path": agree_path
    }
