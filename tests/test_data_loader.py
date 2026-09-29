import pytest
import numpy as np
import pandas as pd
from src.data_loader import raster_to_feature_matrix, feature_matrix_to_raster
from src.preprocessing import validate_feature_schema, FeatureValidationError


def test_raster_and_matrix_roundtrip():
    # 2x2 spatial grid with 3 features
    stack = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ], dtype=np.float32)

    valid_mask = np.array([
        [True, True],
        [False, True]
    ], dtype=bool)

    feature_names = ["f1", "f2", "f3"]
    df, valid_indices = raster_to_feature_matrix(stack, valid_mask, feature_names)

    assert len(df) == 3 # 3 valid pixels
    assert list(df.columns) == feature_names

    # Predict single outcome for valid pixels
    preds = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    grid = feature_matrix_to_raster(preds, valid_indices, (2, 2))

    assert grid[0, 0] == 0.1
    assert grid[0, 1] == 0.5
    assert grid[1, 0] == -9999.0 # masked out NoData
    assert grid[1, 1] == 0.9


def test_feature_order_validation():
    configured = ["featA", "featB", "featC"]
    expected = ["featA", "featB", "featC"]
    assert validate_feature_schema(configured, expected) is True

    mismatch_order = ["featB", "featA", "featC"]
    with pytest.raises(FeatureValidationError):
        validate_feature_schema(mismatch_order, expected)

    mismatch_count = ["featA", "featB"]
    with pytest.raises(FeatureValidationError):
        validate_feature_schema(mismatch_count, expected)
