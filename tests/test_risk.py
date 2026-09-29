import pytest
import numpy as np
from src.risk import normalize_layer, calculate_risk


def test_risk_normalization():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    mask = np.ones_like(arr, dtype=bool)

    norm = normalize_layer(arr, mask, method="minmax")
    assert pytest.approx(norm[0], 1e-4) == 0.0
    assert pytest.approx(norm[-1], 1e-4) == 1.0
    assert pytest.approx(norm[2], 1e-4) == 0.5


def test_risk_calculation_weighted_and_multiplicative():
    s = np.array([0.8], dtype=np.float32)
    e = np.array([0.5], dtype=np.float32)
    v = np.array([0.2], dtype=np.float32)
    mask = np.ones_like(s, dtype=bool)

    prob_w, _ = calculate_risk(s, e, v, mask, method="weighted", weights=(0.4, 0.4, 0.2))
    # 0.4*0.8 + 0.4*0.5 + 0.2*0.2 = 0.32 + 0.20 + 0.04 = 0.56
    assert prob_w[0] == pytest.approx(0.56, abs=1e-4)

    prob_m, _ = calculate_risk(s, e, v, mask, method="multiplicative")
    # 0.8 * 0.5 * 0.2 = 0.08
    assert prob_m[0] == pytest.approx(0.08, abs=1e-4)
