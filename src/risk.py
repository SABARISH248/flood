import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from typing import Dict, Any, List, Tuple, Optional
from src.utils import logger, load_config
from src.predict import write_probability_geotiff
from src.classify import equal_interval_classify


def normalize_layer(
    layer: np.ndarray,
    valid_mask: np.ndarray,
    method: str = "minmax"
) -> np.ndarray:
    """
    Normalize continuous spatial layer array to range [0, 1].
    """
    normalized = np.zeros_like(layer, dtype=np.float32)
    valid_vals = layer[valid_mask]

    if len(valid_vals) == 0:
        return normalized

    if method == "percentile":
        p5, p95 = np.percentile(valid_vals, [5, 95])
        if p95 > p5:
            clipped = np.clip(layer, p5, p95)
            normalized[valid_mask] = (clipped[valid_mask] - p5) / (p95 - p5)
    else: # minmax
        min_v, max_v = float(np.min(valid_vals)), float(np.max(valid_vals))
        if max_v > min_v:
            normalized[valid_mask] = (layer[valid_mask] - min_v) / (max_v - min_v)
        elif 0.0 <= min_v <= 1.0:
            normalized[valid_mask] = layer[valid_mask]
        else:
            normalized[valid_mask] = 0.5

    normalized[~valid_mask] = -9999.0
    return normalized


def calculate_risk(
    susceptibility: np.ndarray,
    exposure: np.ndarray,
    vulnerability: np.ndarray,
    valid_mask: np.ndarray,
    method: str = "weighted",
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate flood risk raster given susceptibility, exposure, and vulnerability layers.
    Methods:
      - 'weighted': Risk = Ws * S + We * E + Wv * V
      - 'multiplicative': Risk = S * E * V
    """
    s_norm = normalize_layer(susceptibility, valid_mask)
    e_norm = normalize_layer(exposure, valid_mask)
    v_norm = normalize_layer(vulnerability, valid_mask)

    risk_prob = np.zeros_like(susceptibility, dtype=np.float32)

    if method == "multiplicative":
        risk_prob[valid_mask] = s_norm[valid_mask] * e_norm[valid_mask] * v_norm[valid_mask]
    else: # weighted
        ws, we, wv = weights
        total_w = ws + we + wv
        ws, we, wv = ws / total_w, we / total_w, wv / total_w
        risk_prob[valid_mask] = (ws * s_norm[valid_mask] + we * e_norm[valid_mask] + wv * v_norm[valid_mask])

    risk_prob[~valid_mask] = -9999.0
    risk_classes, _ = equal_interval_classify(risk_prob, valid_mask)

    return risk_prob, risk_classes


def generate_priority_zones(
    risk_raster: np.ndarray,
    susceptibility_raster: np.ndarray,
    valid_mask: np.ndarray,
    boundaries_gdf: gpd.GeoDataFrame,
    raster_transform: Any,
    raster_crs: Any,
    name_col: str = "NAME_2",
    pop_raster: Optional[np.ndarray] = None,
    buildings_raster: Optional[np.ndarray] = None,
    roads_raster: Optional[np.ndarray] = None,
    hospitals_gdf: Optional[gpd.GeoDataFrame] = None,
    schools_gdf: Optional[gpd.GeoDataFrame] = None
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Aggregate risk raster and exposure metrics to administrative boundaries (districts/blocks/villages).
    Calculates total area, high-risk area, exposed population, exposed buildings, exposed roads, and count of facilities.
    """
    results = []
    # Ensure matching CRS for spatial overlay
    if boundaries_gdf.crs != raster_crs:
        boundaries_gdf = boundaries_gdf.to_crs(raster_crs)

    # Pixel area calculation in sq km (approximate for EPSG:4326 grid or exact transformed)
    res_x = abs(raster_transform.a)
    res_y = abs(raster_transform.e)
    # Approx 111 km per deg latitude if geographical
    pixel_area_km2 = (res_x * 111.0) * (res_y * 111.0) if raster_crs.to_epsg() == 4326 else (res_x * res_y) / 1e6

    # Mask for high risk (risk_prob >= 0.6)
    high_risk_mask = (risk_raster >= 0.6) & valid_mask
    very_high_risk_mask = (risk_raster >= 0.8) & valid_mask

    for idx, row in boundaries_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        unit_name = str(row.get(name_col, f"Unit_{idx}"))
        
        # Create boolean mask for current polygon
        try:
            poly_mask = ~geometry_mask([geom], out_shape=risk_raster.shape, transform=raster_transform, invert=True)
        except Exception:
            continue

        unit_valid = poly_mask & valid_mask
        n_pixels = int(np.sum(unit_valid))
        if n_pixels == 0:
            continue

        total_area_km2 = n_pixels * pixel_area_km2
        high_risk_pixels = int(np.sum(poly_mask & high_risk_mask))
        very_high_risk_pixels = int(np.sum(poly_mask & very_high_risk_mask))

        high_risk_area_km2 = high_risk_pixels * pixel_area_km2
        very_high_risk_area_km2 = very_high_risk_pixels * pixel_area_km2

        mean_s = float(np.mean(susceptibility_raster[unit_valid])) if n_pixels > 0 else 0.0
        mean_r = float(np.mean(risk_raster[unit_valid])) if n_pixels > 0 else 0.0

        # Exposure aggregation
        pop_exp = float(np.sum(pop_raster[poly_mask & high_risk_mask])) if pop_raster is not None else 0.0
        build_exp = int(np.sum(buildings_raster[poly_mask & high_risk_mask])) if buildings_raster is not None else 0
        road_exp_km = float(np.sum(roads_raster[poly_mask & high_risk_mask])) * res_x * 111.0 if roads_raster is not None else 0.0

        # Point facility counts inside administrative boundary
        n_hospitals = 0
        if hospitals_gdf is not None and not hospitals_gdf.empty:
            n_hospitals = int(len(hospitals_gdf[hospitals_gdf.geometry.within(geom)]))

        n_schools = 0
        if schools_gdf is not None and not schools_gdf.empty:
            n_schools = int(len(schools_gdf[schools_gdf.geometry.within(geom)]))

        results.append({
            "name": unit_name,
            "total_area_km2": round(total_area_km2, 2),
            "high_risk_area_km2": round(high_risk_area_km2, 2),
            "very_high_risk_area_km2": round(very_high_risk_area_km2, 2),
            "pct_high_risk": round((high_risk_area_km2 / total_area_km2) * 100, 2) if total_area_km2 > 0 else 0.0,
            "exposed_population": int(max(0, pop_exp)),
            "exposed_buildings": int(max(0, build_exp)),
            "exposed_road_km": round(max(0.0, road_exp_km), 2),
            "hospitals_count": n_hospitals,
            "schools_count": n_schools,
            "mean_susceptibility": round(mean_s, 4),
            "mean_risk": round(mean_r, 4),
            "geometry": geom
        })

    gdf_out = gpd.GeoDataFrame(results, crs=boundaries_gdf.crs)
    df_out = pd.DataFrame(results).drop(columns=["geometry"])
    
    # Sort by estimated exposed population / high risk area descending
    df_out = df_out.sort_values(by="exposed_population", ascending=False).reset_index(drop=True)
    return gdf_out, df_out
