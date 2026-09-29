import os
import yaml
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FloodSense")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def generate_run_id() -> str:
    """Generate a unique run ID for model predictions."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return f"RUN-{timestamp}"


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 checksum for a file."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def save_run_metadata(metadata: Dict[str, Any], output_dir: str = "outputs/reports") -> str:
    """Save execution metadata to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    run_id = metadata.get("run_id", generate_run_id())
    file_path = os.path.join(output_dir, f"{run_id}_metadata.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved run metadata to {file_path}")
    return file_path
