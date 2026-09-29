import pytest
import pandas as pd
import numpy as np
from src.scenarios import run_rainfall_scenario, compare_scenario_impact
from tests.test_predict import DummyModel


def test_scenario_rainfall_modification():
    base_models = {"m1": DummyModel()}
    meta_model = DummyModel()

    df_base = pd.DataFrame({"Rainfall (mm)": [100.0, 200.0], "Temp": [25.0, 30.0]})
    res = run_rainfall_scenario(base_models, meta_model, df_base, rainfall_delta_pct=0.2, rainfall_col="Rainfall (mm)")

    assert res["delta_pct"] == 0.2
    assert "disclaimer" in res
    assert len(res["voting_baseline"]) == 2
    assert len(res["voting_scenario"]) == 2


def test_compare_scenario_impact():
    baseline = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    scenario = np.array([0.2, 0.4, 0.6, 0.8, 0.95], dtype=np.float32)
    mask = np.ones_like(baseline, dtype=bool)
    pop = np.array([100, 200, 300, 400, 500])
    labels = ["Very Low", "Low", "Moderate", "High", "Very High"]

    res = compare_scenario_impact(baseline, scenario, mask, pop, labels)
    assert len(res["class_summary"]) == 5
    assert res["high_risk_population_baseline"] == 900 # 400 + 500
    assert res["high_risk_population_scenario"] == 1200 # 300 + 400 + 500
