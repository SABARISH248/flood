import os
import sys

# Ensure root directory is in sys.path so 'src' and 'api' resolve properly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.utils import load_config
from src.model_registry import ModelRegistry
from api.routes import prediction, risk, feedback

app = FastAPI(
    title="FloodSense India API",
    description="Spatial GIS + ML Flood Risk Prediction & Decision Support System API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(prediction.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "system": "🌊 FloodSense India - Flood Risk Prediction System",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/api/v1/health")
def health_check():
    try:
        config = load_config()
        registry = ModelRegistry()
        meta = registry.get_metadata_summary()

        return {
            "status": "healthy",
            "project_name": config["project"]["name"],
            "version": config["project"]["version"],
            "python_version": sys.version,
            "expected_feature_count": meta["expected_feature_count"],
            "models_status": meta["available_models"]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }
