import os
import sys
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from sklearn.dummy import DummyClassifier
import joblib
# Ensure project root is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import logger, load_config

def generate_placeholder_rasters():
    """Create tiny (10x10) placeholder GeoTIFFs for any raster defined in config.yaml that is missing.
    All rasters will have a simple constant value (e.g., 0.0) and EPSG:4326 CRS.
    """
    cfg = load_config()
    raster_mapping = cfg.get('rasters', {}).get('feature_mapping', {})
    # also vulnerability layers
    vuln_layers = cfg.get('vulnerability', {}).get('layers', [])
    all_paths = list(raster_mapping.values()) + vuln_layers
    for rel_path in all_paths:
        abs_path = os.path.abspath(rel_path)
        if os.path.exists(abs_path):
            continue
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        width = height = 10
        transform = from_origin(0.0, 10.0, 0.01, 0.01)  # arbitrary
        data = np.full((height, width), 0.0, dtype=np.float32)
        meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999.0,
            'width': width,
            'height': height,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform,
        }
        with rasterio.open(abs_path, 'w', **meta) as dst:
            dst.write(data, 1)
        logger.info(f"Created placeholder raster: {abs_path}")

def generate_dummy_model():
    """Create a dummy scikit‑learn model that implements predict_proba.
    The model is saved as `models/dummy.joblib` and registered as a base model.
    """
    cfg = load_config()
    models_dir = os.path.abspath(cfg.get('models', {}).get('directory', 'models/'))
    os.makedirs(models_dir, exist_ok=True)
    dummy = DummyClassifier(strategy='uniform', random_state=0)
    # Fit on minimal data to enable predict_proba
    X = np.zeros((1, 1))
    y = np.array([0])
    dummy.fit(X, y)
    dummy_path = os.path.join(models_dir, 'dummy.joblib')
    joblib.dump(dummy, dummy_path)
    logger.info(f"Created dummy model at {dummy_path}")
    # Update registry.json to include dummy if not present
    registry_path = os.path.join(models_dir, 'registry.json')
    if os.path.exists(registry_path):
        with open(registry_path, 'r', encoding='utf-8') as f:
            reg = json.load(f)
    else:
        reg = {'models': {}, 'feature_names': cfg.get('features', {}).get('exact_order', [])}
    if 'dummy' not in reg['models']:
        reg['models']['dummy'] = {'file': 'dummy.joblib', 'type': 'DummyClassifier'}
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(reg, f, indent=2)
        logger.info("Registered dummy model in registry.json")

if __name__ == '__main__':
    generate_placeholder_rasters()
    generate_dummy_model()
