import os
import json
import joblib
from typing import Dict, Any, List
from src.utils import logger, load_config, compute_file_hash
from src.preprocessing import validate_feature_schema


class ModelRegistry:
    """
    Manages loading, registry metadata, and validation of trained ML models.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.models_dir = self.config["models"]["directory"]
        self.registry_path = os.path.join(self.models_dir, "registry.json")
        self.registry_data = self._load_registry_file()
        self.loaded_models = {}

    def _load_registry_file(self) -> Dict[str, Any]:
        if not os.path.exists(self.registry_path):
            logger.warning(f"Registry file not found at {self.registry_path}. Creating default schema.")
            return {"models": {}, "feature_names": self.config["features"]["exact_order"]}
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_expected_features(self) -> List[str]:
        return self.registry_data.get("feature_names", self.config["features"]["exact_order"])

    def load_model(self, model_key: str):
        """Load a model artifact by key name ('xgboost', 'random_forest', etc.)."""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        models_info = self.registry_data.get("models", {})
        if model_key not in models_info:
            file_name = f"{model_key}.joblib"
        else:
            file_name = models_info[model_key]["file"]

        full_path = os.path.join(self.models_dir, file_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file artifact not found: {full_path}")

        logger.info(f"Loading model '{model_key}' from {full_path}...")
        model = joblib.load(full_path)

        # Inspect feature expectation if model has feature_names_in_
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            configured = self.get_expected_features()
            validate_feature_schema(configured, expected)

        self.loaded_models[model_key] = model
        return model

    def load_all_base_models(self) -> Dict[str, Any]:
        """Load all configured base models."""
        base_keys = self.config["models"]["base_models"]
        models = {}
        for key in base_keys:
            try:
                models[key] = self.load_model(key)
            except Exception as e:
                logger.error(f"Failed to load base model '{key}': {e}")
        return models

    def load_meta_learner(self):
        """Load meta learner for stacking."""
        meta_key = self.config["models"].get("meta_model", "meta_learner")
        return self.load_model(meta_key)

    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get summary of loaded models and registry metadata."""
        summary = {
            "trained_at": self.registry_data.get("trained_at", "Unknown"),
            "expected_feature_count": len(self.get_expected_features()),
            "feature_names": self.get_expected_features(),
            "available_models": []
        }
        for key, info in self.registry_data.get("models", {}).items():
            fpath = os.path.join(self.models_dir, info["file"])
            summary["available_models"].append({
                "name": key,
                "type": info.get("type", "Unknown"),
                "file": info["file"],
                "file_hash": compute_file_hash(fpath),
                "exists": os.path.exists(fpath)
            })
        return summary
