import os
import numpy as np
import pandas as pd
import rasterio
from typing import Dict, List, Any, Tuple, Optional
from src.utils import logger, load_config, generate_run_id, save_run_metadata
from src.data_loader import (
    load_raster_stack,
    raster_to_feature_matrix,
    feature_matrix_to_raster
)
from src.preprocessing import validate_feature_schema
from src.model_registry import ModelRegistry


def predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict positive class (flood) probability for a single model."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1].astype(np.float32)
        return probs.ravel().astype(np.float32)
    elif hasattr(model, "predict"):
        return model.predict(X).astype(np.float32)
    else:
        raise ValueError("Provided model does not support predict_proba or predict interface.")


def predict_probability_all(models: Dict[str, Any], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Generate probability predictions for a dictionary of models."""
    results = {}
    for name, model in models.items():
        results[name] = predict_probability(model, X)
    return results


def soft_voting(probabilities: List[np.ndarray]) -> np.ndarray:
    """
    Paper A Soft Voting Ensemble:
    P_voting(x) = mean(P_model1(x), P_model2(x), ..., P_modelN(x))
    """
    if not probabilities:
        raise ValueError("Probabilities list cannot be empty for soft voting.")
    stacked = np.column_stack(probabilities)
    return np.mean(stacked, axis=1).astype(np.float32)


def stacking_predict(
    base_models: Dict[str, Any],
    meta_model: Any,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Paper B Stacking Ensemble:
    Base predictions -> Meta learner -> Final flood probability.
    """
    base_preds = [predict_probability(m, X) for m in base_models.values()]
    X_meta = np.column_stack(base_preds)
    return predict_probability(meta_model, X_meta)


def write_probability_geotiff(
    raster_grid: np.ndarray,
    target_meta: Dict[str, Any],
    output_path: str
) -> str:
    """Write 2D probability array to a single-band GeoTIFF."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    meta = target_meta.copy()
    meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "nodata": -9999.0
    })
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(raster_grid.astype(np.float32), 1)
    logger.info(f"Saved probability GeoTIFF: {output_path}")
    return output_path


def run_full_inference(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Run end-to-end raster inference for all models (base, soft voting, stacking).
    Saves outputs to outputs/susceptibility/.
    """
    config = load_config(config_path)
    run_id = generate_run_id()
    logger.info(f"Starting full inference run: {run_id}")

    registry = ModelRegistry(config_path)
    expected_features = registry.get_expected_features()

    base_models = registry.load_all_base_models()
    meta_learner = registry.load_meta_learner()

    raster_paths = config["rasters"]["feature_mapping"]
    stack, target_meta, valid_mask = load_raster_stack(raster_paths, expected_features)

    df_features, valid_indices = raster_to_feature_matrix(stack, valid_mask, expected_features)
    validate_feature_schema(list(df_features.columns), expected_features)

    base_probs = predict_probability_all(base_models, df_features)
    voting_probs = soft_voting(list(base_probs.values()))
    stacking_probs = stacking_predict(base_models, meta_learner, df_features)

    all_probs = {**base_probs, "soft_voting": voting_probs, "stacking": stacking_probs}
    output_files = {}

    target_shape = (stack.shape[0], stack.shape[1])
    out_dir = config["outputs"]["susceptibility"]

    for model_name, probs in all_probs.items():
        prob_raster = feature_matrix_to_raster(probs, valid_indices, target_shape)
        out_file = os.path.join(out_dir, f"{model_name}_probability.tif")
        write_probability_geotiff(prob_raster, target_meta, out_file)
        output_files[model_name] = out_file

    meta_info = {
        "run_id": run_id,
        "n_valid_pixels": int(len(valid_indices)),
        "output_files": output_files,
        "models_evaluated": list(all_probs.keys())
    }
    save_run_metadata(meta_info)
    return meta_info


if __name__ == "__main__":
    run_full_inference()
