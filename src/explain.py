import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Tuple, Optional
from sklearn.inspection import permutation_importance
from src.utils import logger


def get_mdi_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    Extract Mean Decrease in Impurity (MDI) feature importances for tree-based estimators.
    """
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        return {feat: round(float(imp), 4) for feat, imp in zip(feature_names, imps)}
    elif hasattr(model, "coef_"):
        # For linear models like LogisticRegression, return absolute normalized coefficients
        coefs = np.abs(model.coef_[0])
        norm_coefs = coefs / np.sum(coefs) if np.sum(coefs) > 0 else coefs
        return {feat: round(float(imp), 4) for feat, imp in zip(feature_names, norm_coefs)}
    else:
        return {feat: 0.0 for feat in feature_names}


def calculate_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Calculate Permutation Feature Importance on a validation dataset.
    """
    res = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc"
    )
    feature_names = list(X_val.columns)
    means = res.importances_mean
    return {feat: round(float(m), 4) for feat, m in zip(feature_names, means)}


def get_shap_explainer(model: Any, background_data: Optional[pd.DataFrame] = None) -> Any:
    """
    Instantiate appropriate SHAP explainer (TreeExplainer or Kernel/Linear Explainer).
    """
    model_type = type(model).__name__
    if any(tree_name in model_type for tree_name in ["RandomForest", "XGB", "LGBM", "DecisionTree"]):
        return shap.TreeExplainer(model)
    else:
        if background_data is not None:
            bg_sample = shap.sample(background_data, min(50, len(background_data)))
            return shap.KernelExplainer(model.predict_proba, bg_sample)
        return shap.Explainer(model)


def calculate_shap_values(
    model: Any,
    X_sample: pd.DataFrame,
    background_data: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values array and base value for input dataframe sample.
    """
    explainer = get_shap_explainer(model, background_data)
    shap_res = explainer(X_sample)

    if hasattr(shap_res, "values"):
        vals = shap_res.values
        base_val = float(shap_res.base_values[0]) if hasattr(shap_res, "base_values") and len(shap_res.base_values) > 0 else 0.5
    else:
        vals = shap_res
        base_val = 0.5

    # If binary classification with 3D output shape (N, M, 2), select class 1
    if vals.ndim == 3 and vals.shape[2] == 2:
        vals = vals[:, :, 1]

    return vals, base_val


def explain_point_prediction(
    model: Any,
    point_features_dict: Dict[str, float],
    feature_names: List[str],
    background_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Provide local SHAP attribution for a single spatial location query.
    Note: SHAP values represent statistical feature contributions to prediction, not causal effects.
    """
    df_point = pd.DataFrame([point_features_dict])[feature_names]
    shap_vals, base_val = calculate_shap_values(model, df_point, background_data)

    point_shap = shap_vals[0]
    contributions = []

    for name, val, shap_v in zip(feature_names, df_point.iloc[0], point_shap):
        contributions.append({
            "feature": name,
            "value": float(val),
            "shap_contribution": round(float(shap_v), 4)
        })

    # Sort features by absolute impact
    contributions = sorted(contributions, key=lambda x: abs(x["shap_contribution"]), reverse=True)

    return {
        "disclaimer": "These features contributed most to this model prediction (statistical attribution, not causal effect).",
        "base_value": round(base_val, 4),
        "contributions": contributions
    }
