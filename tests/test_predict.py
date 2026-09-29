import pytest
import numpy as np
import pandas as pd
from src.predict import soft_voting, stacking_predict


def test_soft_voting_basic():
    m1 = np.array([0.2, 0.8], dtype=np.float32)
    m2 = np.array([0.4, 0.6], dtype=np.float32)
    m3 = np.array([0.6, 0.4], dtype=np.float32)

    res = soft_voting([m1, m2, m3])
    assert len(res) == 2
    assert pytest.approx(res[0], 1e-4) == 0.4
    assert pytest.approx(res[1], 1e-4) == 0.6


def test_soft_voting_empty():
    with pytest.raises(ValueError):
        soft_voting([])


class DummyModel:
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            vals = X.iloc[:, 0].values
        else:
            vals = X[:, 0]
        return np.column_stack([1 - vals, vals])


def test_stacking_predict():
    base_models = {
        "m1": DummyModel(),
        "m2": DummyModel()
    }
    meta_model = DummyModel()

    df = pd.DataFrame({"feat1": [0.3, 0.7]})
    res = stacking_predict(base_models, meta_model, df)
    assert len(res) == 2
    assert res.ndim == 1
