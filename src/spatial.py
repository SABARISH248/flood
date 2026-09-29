import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import KFold
from src.utils import logger
from src.validate import evaluate_predictions


def create_spatial_blocks(
    gdf: gpd.GeoDataFrame,
    block_size_deg: float = 0.25
) -> np.ndarray:
    """
    Assign a spatial block ID to each point in GeoDataFrame based on a grid of size block_size_deg.
    """
    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()

    block_x = np.floor((coords[:, 0] - min_x) / block_size_deg).astype(int)
    block_y = np.floor((coords[:, 1] - min_y) / block_size_deg).astype(int)

    # Combine x and y block indices into a unique integer block ID
    block_ids = block_x * 1000 + block_y
    return block_ids


def spatial_block_cv(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    gdf: gpd.GeoDataFrame,
    n_splits: int = 5,
    block_size_deg: float = 0.25
) -> Dict[str, Any]:
    """
    Perform Spatial Block Cross-Validation to evaluate model performance without spatial leakage.
    """
    blocks = create_spatial_blocks(gdf, block_size_deg=block_size_deg)
    unique_blocks = np.unique(blocks)

    kf = KFold(n_splits=min(n_splits, len(unique_blocks)), shuffle=True, random_state=42)
    auc_scores, f1_scores = [], []

    for train_block_idx, test_block_idx in kf.split(unique_blocks):
        test_block_ids = unique_blocks[test_block_idx]
        test_mask = np.isin(blocks, test_block_ids)
        train_mask = ~test_mask

        if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
            continue
        if len(np.unique(y[test_mask])) < 2 or len(np.unique(y[train_mask])) < 2:
            continue

        X_tr, y_tr = X.iloc[train_mask], y[train_mask]
        X_te, y_te = X.iloc[test_mask], y[test_mask]

        model.fit(X_tr, y_tr)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_te)[:, 1]
        else:
            probs = model.predict(X_te)

        res = evaluate_predictions(y_te, probs)
        auc_scores.append(res["roc_auc"])
        f1_scores.append(res["f1_score"])

    return {
        "spatial_cv_auc_mean": round(float(np.mean(auc_scores)), 4) if auc_scores else 0.0,
        "spatial_cv_auc_std": round(float(np.std(auc_scores)), 4) if auc_scores else 0.0,
        "spatial_cv_f1_mean": round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
        "n_blocks": int(len(unique_blocks))
    }


def leave_one_flood_year_out_cv(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    years: np.ndarray
) -> Dict[str, Any]:
    """
    Perform Leave-One-Flood-Year-Out validation across available historical flood years.
    """
    unique_years = np.unique(years[~pd.isna(years)])
    if len(unique_years) <= 1:
        logger.warning("Fewer than 2 flood years available. Cannot execute Leave-One-Year-Out CV.")
        return {"year_out_auc_mean": 0.0, "evaluated_years": list(unique_years)}

    auc_scores, f1_scores = [], []
    year_results = {}

    for yr in unique_years:
        test_mask = (years == yr)
        train_mask = ~test_mask

        if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
            continue
        if len(np.unique(y[test_mask])) < 2 or len(np.unique(y[train_mask])) < 2:
            continue

        X_tr, y_tr = X.iloc[train_mask], y[train_mask]
        X_te, y_te = X.iloc[test_mask], y[test_mask]

        model.fit(X_tr, y_tr)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_te)[:, 1]
        else:
            probs = model.predict(X_te)

        res = evaluate_predictions(y_te, probs)
        auc_scores.append(res["roc_auc"])
        f1_scores.append(res["f1_score"])
        year_results[str(yr)] = res

    return {
        "year_out_auc_mean": round(float(np.mean(auc_scores)), 4) if auc_scores else 0.0,
        "year_out_f1_mean": round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
        "per_year_results": year_results
    }


def generate_paper_b_non_flood_samples(
    flood_points: gpd.GeoDataFrame,
    study_extent_bounds: Tuple[float, float, float, float],
    streams_gdf: Optional[gpd.GeoDataFrame] = None,
    case_i_dist_km: float = 5.0,
    case_ii_dist_km: float = 3.0,
    case_ii_stream_dist_km: float = 4.0,
    n_samples: int = 1000
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Paper B Non-Flood Point Sampling Strategy:
    Case I: Non-flood samples >= case_i_dist_km (5 km) from flood points.
    Case II: Non-flood samples >= case_ii_dist_km (3 km) from flood points AND >= case_ii_stream_dist_km (4 km) from streams.
    """
    minx, miny, maxx, maxy = study_extent_bounds

    # Approximate 1 degree ~ 111 km for buffer distance conversion in lat/lon
    deg_per_km = 1.0 / 111.0

    buffer_case_i = flood_points.geometry.buffer(case_i_dist_km * deg_per_km).unary_union
    buffer_case_ii_flood = flood_points.geometry.buffer(case_ii_dist_km * deg_per_km).unary_union

    if streams_gdf is not None and not streams_gdf.empty:
        buffer_case_ii_stream = streams_gdf.geometry.buffer(case_ii_stream_dist_km * deg_per_km).unary_union
        buffer_case_ii = buffer_case_ii_flood.union(buffer_case_ii_stream)
    else:
        buffer_case_ii = buffer_case_ii_flood

    np.random.seed(42)
    pts_case_i = []
    pts_case_ii = []

    attempts = 0
    max_attempts = n_samples * 50

    while (len(pts_case_i) < n_samples or len(pts_case_ii) < n_samples) and attempts < max_attempts:
        attempts += 1
        rx = np.random.uniform(minx, maxx)
        ry = np.random.uniform(miny, maxy)
        pt = Point(rx, ry)

        if len(pts_case_i) < n_samples and not buffer_case_i.contains(pt):
            pts_case_i.append(pt)

        if len(pts_case_ii) < n_samples and not buffer_case_ii.contains(pt):
            pts_case_ii.append(pt)

    gdf_case_i = gpd.GeoDataFrame(geometry=pts_case_i, crs="EPSG:4326")
    gdf_case_ii = gpd.GeoDataFrame(geometry=pts_case_ii, crs="EPSG:4326")

    logger.info(f"Sampled Case I non-flood points: {len(gdf_case_i)}, Case II points: {len(gdf_case_ii)}")
    return gdf_case_i, gdf_case_ii
