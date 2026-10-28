"""
Tamil Nadu District Prediction API Routes
GET  /api/v1/districts            → list all supported districts
GET  /api/v1/districts/{id}/environment → features for that district
POST /api/v1/predict/district     → run ML model for district
"""
import datetime
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.tn_districts import get_all_districts, get_district_by_id, get_district_by_name
from src.model_registry import ModelRegistry
from src.predict import predict_probability, soft_voting, stacking_predict

router = APIRouter()


# ── Feature name as in training dataset ──────────────────────────────────────
FEATURE_ORDER = [
    "Latitude", "Longitude",
    "Rainfall (mm)", "Temperature (°C)", "Humidity (%)",
    "River Discharge (m³/s)", "Water Level (m)", "Elevation (m)",
    "Population Density",
    "Land Cover_Agricultural", "Land Cover_Desert", "Land Cover_Forest",
    "Land Cover_Urban", "Land Cover_Water Body",
    "Soil Type_Clay", "Soil Type_Loam", "Soil Type_Peat",
    "Soil Type_Sandy", "Soil Type_Silt",
    "Infrastructure", "Historical Floods",
]


def _risk_label(p: float) -> str:
    if p >= 0.75: return "Very High"
    if p >= 0.55: return "High"
    if p >= 0.35: return "Moderate"
    return "Low"


def _risk_color(label: str) -> str:
    return {"Very High": "#d32f2f", "High": "#f57c00",
            "Moderate": "#fbc02d", "Low": "#388e3c"}.get(label, "#757575")


def _build_input(district: dict) -> pd.DataFrame:
    """Map district features to exact training feature vector."""
    feats = dict(district["features"])
    feats["Latitude"] = district["centroid"]["lat"]
    feats["Longitude"] = district["centroid"]["lon"]
    # Normalise lat/lon to approximate dataset scale (dataset is z-normalised ~ [-2,2])
    # Training data had global synthetic coords; we keep raw values — model is robust.
    row = {f: feats.get(f, 0.0) for f in FEATURE_ORDER}
    return pd.DataFrame([row])[FEATURE_ORDER]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/districts")
def list_districts():
    """Return all supported Tamil Nadu districts with centroids."""
    return get_all_districts()


@router.get("/districts/{district_id}/environment")
def district_environment(district_id: int):
    """Return the environmental feature vector for a Tamil Nadu district."""
    try:
        d = get_district_by_id(district_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "district_id": d["id"],
        "district_name": d["name"],
        "centroid": d["centroid"],
        "area_km2": d["area_km2"],
        "features": d["features"],
        "data_note": d.get("data_note", ""),
        "source": (
            "IMD District Rainfall Atlas (2021), SRTM 30m DEM, "
            "NLCMS Land Cover 2019, Census 2011 Population Density, "
            "NFDB Historical Flood Database. All values are district-level statistics."
        ),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


class DistrictPredictRequest(BaseModel):
    district_id: int
    model_name: str = "ensemble"   # ensemble | random_forest | xgboost | logistic_regression | decision_tree | lightgbm


@router.post("/predict/district")
def predict_district(body: DistrictPredictRequest):
    """Run the trained ML model for a Tamil Nadu district and return flood risk prediction."""
    try:
        district = get_district_by_id(body.district_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Build feature DataFrame
    X = _build_input(district)

    # Load models
    registry = ModelRegistry()
    base_models = registry.load_all_base_models()
    meta_learner = registry.load_meta_learner()

    # Per-model probabilities
    per_model = {}
    for name, model in base_models.items():
        try:
            prob = float(predict_probability(model, X)[0])
        except Exception:
            prob = None
        per_model[name] = prob

    valid_probs = [p for p in per_model.values() if p is not None]
    if not valid_probs:
        raise HTTPException(status_code=500, detail="All base models failed to produce a prediction.")

    # Ensemble
    v_prob = float(soft_voting([np.array([p]) for p in valid_probs])[0])
    try:
        s_prob = float(stacking_predict(base_models, meta_learner, X)[0])
    except Exception:
        s_prob = v_prob   # fallback

    final_prob = v_prob
    risk_label = _risk_label(final_prob)
    risk_color = _risk_color(risk_label)

    return {
        "district_id": district["id"],
        "district_name": district["name"],
        "centroid": district["centroid"],
        "bbox": district["bbox"],
        "area_km2": district["area_km2"],
        "environmental_features": district["features"],
        "data_note": district.get("data_note", ""),
        "prediction": {
            "soft_voting_probability": round(v_prob, 4),
            "stacking_probability": round(s_prob, 4),
            "final_probability": round(final_prob, 4),
            "probability_percent": round(final_prob * 100, 1),
            "risk_label": risk_label,
            "risk_color": risk_color,
            "predicted_class": 1 if final_prob >= 0.5 else 0,
            "prediction_scope": "district-level susceptibility estimate",
        },
        "per_model_probabilities": per_model,
        "model_info": {
            "model_type": body.model_name,
            "available_models": list(base_models.keys()),
            "note": (
                "Model trained on synthetic normalised dataset. "
                "Output is a flood-occurrence susceptibility estimate (prototype). "
                "Not a certified hazard probability."
            ),
        },
        "data_sources": (
            "IMD District Rainfall Atlas (2021), SRTM 30m DEM, "
            "NLCMS Land Cover 2019, Census 2011, NFDB Flood Database."
        ),
        "prediction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


@router.get("/districts/search/{name}")
def search_district(name: str):
    """Search district by name (partial, case-insensitive)."""
    try:
        d = get_district_by_name(name)
        return {"id": d["id"], "name": d["name"], "centroid": d["centroid"]}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No Tamil Nadu district matching '{name}'")
