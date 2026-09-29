import pandas as pd
from typing import List, Dict, Any
from src.utils import logger


class FeatureValidationError(Exception):
    """Raised when feature order or feature count does not match expected model schema."""
    pass


def validate_feature_schema(
    input_features: List[str],
    expected_features: List[str]
) -> bool:
    """
    Validate configured/input feature list against trained model expected feature order.
    Normalizes spaces to underscores to support models like LightGBM that sanitize feature names.
    Throws FeatureValidationError if count or exact order differs.
    """
    if len(input_features) != len(expected_features):
        msg = f"Feature count mismatch! Expected {len(expected_features)} features, but received {len(input_features)}."
        logger.error(msg)
        raise FeatureValidationError(msg)

    mismatches = []
    for idx, (inp, exp) in enumerate(zip(input_features, expected_features)):
        norm_inp = str(inp).replace(" ", "_")
        norm_exp = str(exp).replace(" ", "_")
        if norm_inp != norm_exp:
            mismatches.append(f"Position {idx}: Input '{inp}' vs Expected '{exp}'")

    if mismatches:
        msg = f"Feature order mismatch detected:\n" + "\n".join(mismatches)
        logger.error(msg)
        raise FeatureValidationError(msg)

    logger.info("Feature schema validation passed successfully!")
    return True


def preprocess_feature_df(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """
    Validate and return feature dataframe matching exact expected feature schema.
    """
    validate_feature_schema(list(df.columns), expected_features)
    return df[expected_features]
