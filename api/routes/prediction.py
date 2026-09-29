from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.utils import load_config
from src.data_loader import sample_raster_at_point
from src.model_registry import ModelRegistry
from src.predict import predict_probability, soft_voting, stacking_predict, run_full_inference
from src.explain import explain_point_prediction
from src.classify import equal_interval_classify

router = APIRouter(prefix="/predict", tags=["Prediction"])


class PointPredictionRequest(BaseModel):
    latitude: float = Field(..., example=25.5941)
    longitude: float = Field(..., example=85.1376)


class PointPredictionResponse(BaseModel):
    latitude: float
    longitude: float
    features: Dict[str, float]
    base_model_probabilities: Dict[str, float]
    ensemble_probabilities: Dict[str, float]
    susceptibility_class: str
    risk_class: str
    explanation: Dict[str, Any]


@router.post("/point", response_model=PointPredictionResponse)
def predict_at_point(req: PointPredictionRequest):
    try:
        config = load_config()
        registry = ModelRegistry()
        features_order = registry.get_expected_features()
        raster_paths = config["rasters"]["feature_mapping"]

        # Sample feature raster values at point location
        point_features = sample_raster_at_point(raster_paths, features_order, req.latitude, req.longitude)
        df_point = pd.DataFrame([point_features])[features_order]

        base_models = registry.load_all_base_models()
        meta_learner = registry.load_meta_learner()

        base_probs = {}
        for name, model in base_models.items():
            base_probs[name] = round(float(predict_probability(model, df_point)[0]), 4)

        v_prob = round(float(soft_voting([np.array([p]) for p in base_probs.values()])[0]), 4)
        s_prob = round(float(stacking_predict(base_models, meta_learner, df_point)[0]), 4)

        ensemble_probs = {
            "soft_voting": v_prob,
            "stacking": s_prob
        }

        # Determine class label
        prob_val = v_prob
        if prob_val < 0.2:
            s_class = "Very Low"
        elif prob_val < 0.4:
            s_class = "Low"
        elif prob_val < 0.6:
            s_class = "Moderate"
        elif prob_val < 0.8:
            s_class = "High"
        else:
            s_class = "Very High"

        # Explanation via XGBoost if available
        xgb_model = base_models.get("xgboost", list(base_models.values())[0])
        explanation = explain_point_prediction(xgb_model, point_features, features_order)

        return {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "features": point_features,
            "base_model_probabilities": base_probs,
            "ensemble_probabilities": ensemble_probs,
            "susceptibility_class": s_class,
            "risk_class": s_class,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/raster")
def trigger_full_raster_inference():
    try:
        res = run_full_inference()
        return {"status": "success", "run_metadata": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
