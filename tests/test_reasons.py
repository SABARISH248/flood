import numpy as np
from src.reasons import explain_drivers, compute_background_stats, decompose_risk, percentile_of


def test_explain_drivers_ranking_and_text():
    feat_vals = {"Rainfall (mm)": 1800.0, "Elevation (m)": 45.0, "Humidity (%)": 80.0}
    imps = {"Rainfall (mm)": 0.31, "Elevation (m)": 0.24, "Humidity (%)": -0.05}
    bg = {
        "Rainfall (mm)": {"median": 1000.0, "p10": 500.0, "p25": 750.0, "p75": 1300.0,
                          "p90": 1600.0, "min": 200.0, "max": 2500.0, "n": 1000},
        "Elevation (m)": {"median": 200.0, "p10": 50.0, "p25": 100.0, "p75": 400.0,
                          "p90": 700.0, "min": 10.0, "max": 1200.0, "n": 1000},
    }
    drivers = explain_drivers(feat_vals, imps, bg, top_n=2)
    assert len(drivers) == 2
    assert drivers[0]["feature"] == "Rainfall (mm)"  # highest |importance|
    assert "percentile" in drivers[0] and drivers[0]["percentile"] > 75
    assert "pushes risk up" in drivers[0]["text"]
    assert "caus" not in drivers[0]["text"].lower()


def test_background_stats_and_percentile():
    rasters = {"Rainfall (mm)": np.linspace(0, 100, 1000).reshape(10, 100)}
    stats = compute_background_stats(rasters, ["Rainfall (mm)"])
    assert abs(stats["Rainfall (mm)"]["median"] - 50.0) < 1.0
    assert abs(percentile_of(75.0, stats["Rainfall (mm)"]) - 75.0) < 2.0


def test_decompose_risk_weighted():
    out = decompose_risk(0.8, 0.5, 0.4,
                         weights={"susceptibility": 0.4, "exposure": 0.4, "vulnerability": 0.2})
    assert out["dominant"] == "susceptibility"
    assert abs(sum(out["shares"].values()) - 1.0) < 1e-6
