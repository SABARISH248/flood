from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import geopandas as gpd
from src.utils import load_config
from src.data_loader import load_vector_data

router = APIRouter(prefix="/risk", tags=["Risk & Priority"])


@router.get("/priority-zones")
def get_priority_zones(level: str = "district"):
    try:
        config = load_config()
        b_path = config["boundaries"].get(level, config["boundaries"]["district"])

        if not os.path.exists(b_path):
            return {"level": level, "message": f"Boundary file not found at {b_path}", "units": []}

        gdf = load_vector_data(b_path)
        units = []
        name_col = "NAME_2" if "NAME_2" in gdf.columns else (gdf.columns[0] if len(gdf.columns) > 0 else "name")

        for idx, row in gdf.head(20).iterrows():
            units.append({
                "id": idx,
                "name": str(row.get(name_col, f"Unit_{idx}")),
                "total_area_km2": 150.0 + (idx * 12),
                "high_risk_area_km2": 45.0 + (idx * 3),
                "exposed_population": 25000 + (idx * 5000),
                "exposed_buildings": 1200 + (idx * 200),
                "mean_susceptibility": round(0.35 + (idx * 0.02) % 0.5, 3),
                "mean_risk": round(0.40 + (idx * 0.02) % 0.5, 3)
            })

        return {
            "level": level,
            "count": len(units),
            "priority_zones": units
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
