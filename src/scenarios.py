import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from src.utils import logger
from src.predict import predict_probability, soft_voting, stacking_predict
from src.classify import equal_interval_classify


SCENARIO_DISCLAIMER = "This is a sensitivity scenario, not a calibrated real-time flood forecast."


def run_rainfall_scenario(
    base_models: Dict[str, Any],
    meta_model: Any,
    df_features_baseline: pd.DataFrame,
    rainfall_delta_pct: float,
    rainfall_col: str = "Rainfall (mm)"
) -> Dict[str, Any]:
    """
    Run what-if scenario by modifying rainfall predictor values by rainfall_delta_pct (e.g. +20% -> 0.2).
    Calculates scenario probability, baseline probability, and probability difference.
    """
    if rainfall_col not in df_features_baseline.columns:
        logger.warning(f"Rainfall column '{rainfall_col}' not found in feature matrix.")
        df_scenario = df_features_baseline.copy()
    else:
        df_scenario = df_features_baseline.copy()
        df_scenario[rainfall_col] = df_scenario[rainfall_col] * (1.0 + rainfall_delta_pct)

    # Base predictions baseline vs scenario
    base_probs_base = [predict_probability(m, df_features_baseline) for m in base_models.values()]
    base_probs_scen = [predict_probability(m, df_scenario) for m in base_models.values()]

    voting_base = soft_voting(base_probs_base)
    voting_scen = soft_voting(base_probs_scen)

    stacking_base = stacking_predict(base_models, meta_model, df_features_baseline)
    stacking_scen = stacking_predict(base_models, meta_model, df_scenario)

    diff_prob = voting_scen - voting_base

    return {
        "disclaimer": SCENARIO_DISCLAIMER,
        "delta_pct": rainfall_delta_pct,
        "voting_baseline": voting_base,
        "voting_scenario": voting_scen,
        "stacking_baseline": stacking_base,
        "stacking_scenario": stacking_scen,
        "diff_prob": diff_prob
    }


def compare_scenario_impact(
    baseline_probs: np.ndarray,
    scenario_probs: np.ndarray,
    valid_mask: np.ndarray,
    pop_matrix: np.ndarray,
    class_labels: List[str]
) -> Dict[str, Any]:
    """
    Calculate class area changes and population shift between baseline and scenario predictions.
    """
    cls_base, _ = equal_interval_classify(baseline_probs, valid_mask)
    cls_scen, _ = equal_interval_classify(scenario_probs, valid_mask)

    total_valid = np.sum(valid_mask)
    class_summary = []

    for c in range(1, 6):
        lbl = class_labels[c - 1] if c - 1 < len(class_labels) else f"Class {c}"
        n_base = np.sum((cls_base == c) & valid_mask)
        n_scen = np.sum((cls_scen == c) & valid_mask)

        pct_base = (n_base / total_valid * 100.0) if total_valid > 0 else 0.0
        pct_scen = (n_scen / total_valid * 100.0) if total_valid > 0 else 0.0
        delta_pct = pct_scen - pct_base

        class_summary.append({
            "class": lbl,
            "baseline_pct": round(pct_base, 2),
            "scenario_pct": round(pct_scen, 2),
            "delta_pct": round(delta_pct, 2)
        })

    # Population in high risk (classes 4 and 5: High and Very High)
    high_base = (cls_base >= 4) & valid_mask
    high_scen = (cls_scen >= 4) & valid_mask

    pop_high_base = float(np.sum(pop_matrix[high_base])) if pop_matrix is not None else 0.0
    pop_high_scen = float(np.sum(pop_matrix[high_scen])) if pop_matrix is not None else 0.0

    return {
        "class_summary": class_summary,
        "high_risk_population_baseline": int(pop_high_base),
        "high_risk_population_scenario": int(pop_high_scen),
        "high_risk_population_delta": int(pop_high_scen - pop_high_base)
    }
