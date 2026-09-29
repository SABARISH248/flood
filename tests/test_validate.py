import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from affine import Affine
from src.validate import evaluate_predictions, calculate_prediction_rate


def test_evaluate_predictions_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.8, 0.9])

    res = evaluate_predictions(y_true, y_prob)
    assert res["roc_auc"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1_score"] == 1.0


def test_prediction_rate_formula():
    classified_raster = np.array([
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [5, 5, 5, 5]
    ], dtype=np.int32)
    valid_mask = np.ones_like(classified_raster, dtype=bool)

    transform = Affine.identity()

    # 2 flood points located in class 5 pixels
    pts = [Point(0, 2), Point(1, 2)]
    flood_gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")

    res = calculate_prediction_rate(classified_raster, valid_mask, flood_gdf, transform)

    # Class 5 has 4 pixels out of 12 = 33.33% area
    # Observed points in class 5 = 2/2 = 100%
    # Prediction rate for Class 5 = 100 / 33.33 ~ 3.0
    assert "Very High" in res
    assert res["Very High"]["observed_points"] == 2
    assert res["Very High"]["prediction_rate"] > 2.5
