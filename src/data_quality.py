import os
import json
from typing import Dict, Any

from src.utils import logger, load_config

def check_file_exists(path: str) -> bool:
    """Return True if the given path exists on disk."""
    exists = os.path.exists(path)
    if not exists:
        logger.warning(f"Missing required file: {path}")
    return exists

def run_data_quality_check(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Inspect the repository for required raster, vector, and model files.

    Returns a dictionary with categories (rasters, vectors, models) and a
    per‑item boolean indicating presence. The function is lightweight and can be
    called from the API health endpoint or a Streamlit page.
    """
    cfg = load_config(config_path)
    report: Dict[str, Any] = {
        "rasters": {},
        "vectors": {},
        "models": {},
        "boundaries": {},
        "exposure": {},
        "inventory": {},
    }

    # Rasters defined under cfg["rasters"]["feature_mapping"]
    raster_mapping = cfg.get("rasters", {}).get("feature_mapping", {})
    for name, rel_path in raster_mapping.items():
        full_path = os.path.abspath(rel_path)
        report["rasters"][name] = check_file_exists(full_path)

    # Vulnerability rasters list
    for vuln_path in cfg.get("vulnerability", {}).get("layers", []):
        full_path = os.path.abspath(vuln_path)
        report["rasters"][os.path.basename(vuln_path)] = check_file_exists(full_path)

    # Vectors / inventory
    inventory_path = cfg.get("inventory", {}).get("path")
    if inventory_path:
        report["inventory"]["flood_inventory"] = check_file_exists(os.path.abspath(inventory_path))

    # Boundaries
    for level, path in cfg.get("boundaries", {}).items():
        report["boundaries"][level] = check_file_exists(os.path.abspath(path))

    # Exposure rasters / vectors
    exposure_cfg = cfg.get("exposure", {})
    for key, path in exposure_cfg.items():
        if isinstance(path, str):
            report["exposure"][key] = check_file_exists(os.path.abspath(path))
        elif isinstance(path, dict):
            # ignore potential nested dicts
            pass

    # Models (registry will also report existence, but we check the files directly)
    models_dir = cfg.get("models", {}).get("directory", "models/")
    base_keys = cfg.get("models", {}).get("base_models", [])
    for key in base_keys:
        model_path = os.path.join(models_dir, f"{key}.joblib")
        report["models"][key] = check_file_exists(os.path.abspath(model_path))
    # meta model
    meta_key = cfg.get("models", {}).get("meta_model", "meta_learner")
    meta_path = os.path.join(models_dir, f"{meta_key}.joblib")
    report["models"][meta_key] = check_file_exists(os.path.abspath(meta_path))

    return report
