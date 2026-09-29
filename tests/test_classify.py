import pytest
import numpy as np
from src.classify import equal_interval_classify, jenks_classify


def test_equal_interval_classify():
    raster = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    mask = np.ones_like(raster, dtype=bool)

    classified, breaks = equal_interval_classify(raster, mask)
    assert np.array_equal(classified, np.array([1, 2, 3, 4, 5]))
    assert breaks == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_jenks_classify_class_count():
    raster = np.linspace(0.01, 0.99, 100, dtype=np.float32)
    mask = np.ones_like(raster, dtype=bool)

    classified, breaks = jenks_classify(raster, mask, n_classes=5)
    unique_classes = np.unique(classified)

    assert len(breaks) == 6 # 5 classes means 6 break points
    assert len(unique_classes) == 5
    assert np.all(classified >= 1) and np.all(classified <= 5)
