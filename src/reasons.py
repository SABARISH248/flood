"""Plain-language explanations of why predicted risk is high/low at a point.

Wording is deliberately non-causal: features 'contributed to this model
prediction'. Never 'caused the flood'.
"""

import numpy as np
from typing import Dict, List, Optional, Any

HIGH_PCT = 75.0
LOW_PCT = 25.0

# feature -> (pretty label, unit, higher_is_riskier or None if unknown/categorical)
# Keys match config.yaml features.exact_order / models/registry.json.
FEATURE_INFO = {
    "Latitude": ("Latitude", "°N", None),
    "Longitude": ("Longitude", "°E", None),
    "Rainfall (mm)": ("Rainfall", "mm", True),
    "Temperature (°C)": ("Temperature", "°C", None),
    "Humidity (%)": ("Humidity", "%", True),
    "River Discharge (m³/s)": ("River discharge", "m³/s", True),
    "Water Level (m)": ("Water level", "m", True),
    "Elevation (m)": ("Elevation", "m", False),
    "Population Density": ("Population density", "persons/km²", True),
    "Land Cover_Agricultural": ("Cropland cover", "fraction", None),
    "Land Cover_Desert": ("Desert/barren cover", "fraction", None),
    "Land Cover_Forest": ("Forest cover", "fraction", False),
    "Land Cover_Urban": ("Urban cover", "fraction", True),
    "Land Cover_Water Body": ("Water-body cover", "fraction", True),
    "Soil Type_Clay": ("Clay soil", "fraction", True),
    "Soil Type_Loam": ("Loam soil", "fraction", None),
    "Soil Type_Peat": ("Peat soil", "fraction", True),
    "Soil Type_Sandy": ("Sandy soil", "fraction", False),
    "Soil Type_Silt": ("Silt soil", "fraction", None),
    "Infrastructure": ("Infrastructure index", "", True),
    "Historical Floods": ("Historical flood frequency", "", True),
    # legacy snake_case aliases (other configs)
    "rainfall": ("Rainfall", "mm", True),
    "elevation": ("Elevation", "m", False),
    "slope": ("Slope", "deg", False),
    "distance_to_river": ("Distance to river", "km", False),
    "twi": ("Wetness index (TWI)", "", True),
    "ndvi": ("Vegetation (NDVI)", "", False),
}


def compute_background_stats(raster_dict: Dict[str, np.ndarray],
                             feature_order: List[str],
                             max_samples: int = 200000,
                             seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Percentile background per feature from the raster stack (for 'high/low vs typical')."""
    rng = np.random.default_rng(seed)
    stats = {}
    for feat in feature_order:
        if feat not in raster_dict:
            continue
        v = np.asarray(raster_dict[feat], dtype=float).ravel()
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        if len(v) > max_samples:
            v = rng.choice(v, max_samples, replace=False)
        stats[feat] = {
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "n": int(len(v)),
        }
    return stats


def percentile_of(value: float, feat_stats: Dict[str, float]) -> float:
    """Approximate percentile of value within background (linear interp of stored quantiles)."""
    xs = [feat_stats["min"], feat_stats["p10"], feat_stats["p25"],
          feat_stats["median"], feat_stats["p75"], feat_stats["p90"], feat_stats["max"]]
    ps = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]
    return float(np.interp(value, xs, ps))


def explain_drivers(feature_values: Dict[str, float],
                    importances: Dict[str, float],
                    background_stats: Optional[Dict[str, Dict[str, float]]] = None,
                    top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Rank drivers of a point prediction and phrase them in plain language.

    Parameters
    ----------
    feature_values : dict
        {feature: value at point}.
    importances : dict
        {feature: weight}; magnitude = influence, sign>0 pushes toward flood.
    background_stats : dict, optional
        Output of compute_background_stats; enables high/low-vs-typical phrasing.
    top_n : int
        Number of drivers to return.

    Returns
    -------
    list of {feature, label, value, importance, percentile, level, direction, text}
    """
    ranked = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
    drivers = []
    for feat, imp in ranked[:top_n]:
        if feat not in feature_values or feature_values[feat] is None:
            continue
        val = float(feature_values[feat])
        label, unit, higher_riskier = FEATURE_INFO.get(feat, (feat.replace("_", " ").title(), "", None))
        unit_s = f" {unit}" if unit else ""

        pct, level = None, "typical"
        if background_stats and feat in background_stats:
            pct = percentile_of(val, background_stats[feat])
            level = "high" if pct >= HIGH_PCT else ("low" if pct <= LOW_PCT else "typical")

        pushes_up = (imp > 0)
        if higher_riskier is True:
            direction = "pushes risk up" if (level == "high") == pushes_up or (level == "typical" and pushes_up) else "pushes risk down"
            assoc = f"higher {label.lower()} is associated with higher flood susceptibility in this model"
        elif higher_riskier is False:
            direction = "pushes risk up" if (level == "low") == pushes_up or (level == "typical" and pushes_up) else "pushes risk down"
            assoc = f"lower {label.lower()} is associated with higher flood susceptibility in this model"
        else:
            direction = "pushes risk up" if pushes_up else "pushes risk down"
            assoc = "this category/value is associated with higher flood susceptibility in this model" if pushes_up else \
                    "this category/value is associated with lower flood susceptibility in this model"

        pct_s = f" ({pct:.0f}th percentile — {level} for the study area)" if pct is not None else ""
        text = f"{label} is {val:.3g}{unit_s}{pct_s}; {assoc} — {direction}."
        drivers.append({
            "feature": feat, "label": label, "value": val,
            "importance": float(imp), "percentile": pct, "level": level,
            "direction": direction, "text": text,
        })
    return drivers


def decompose_risk(susceptibility: float, exposure: float, vulnerability: float,
                   weights: Optional[Dict[str, float]] = None,
                   method: str = "weighted") -> Dict[str, Any]:
    """Show which of S/E/V drives the risk score (shares sum to ~1 for weighted)."""
    if weights is None:
        weights = {"susceptibility": 0.4, "exposure": 0.4, "vulnerability": 0.2}
    S, E, V = float(susceptibility), float(exposure), float(vulnerability)
    if method == "multiplicative":
        parts = {"susceptibility": S, "exposure": E, "vulnerability": V}
        total = S * E * V
        # log-share: each factor's proportional drag on the product
        if total > 0 and all(v > 0 for v in parts.values()):
            log_total = np.log(total)
            shares = {k: float(np.log(v) / log_total) if log_total != 0 else 1 / 3 for k, v in parts.items()}
        else:
            shares = {k: 1 / 3 for k in parts}
    else:
        parts = {"susceptibility": weights["susceptibility"] * S,
                 "exposure": weights["exposure"] * E,
                 "vulnerability": weights["vulnerability"] * V}
        total = sum(parts.values())
        shares = {k: float(v / total) if total > 0 else 1 / 3 for k, v in parts.items()}
    top = max(shares, key=shares.get)
    return {"components": {"susceptibility": S, "exposure": E, "vulnerability": V},
            "contributions": parts, "shares": shares,
            "dominant": top,
            "summary": f"Risk is driven mainly by {top} ({shares[top] * 100:.0f}% of the {method} score)."}
