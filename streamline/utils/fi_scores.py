from __future__ import annotations

from typing import Sequence
import numpy as np


def build_ordered_score_vector(
    feature_names: Sequence[str],
    score_dict: dict[str, float],
    *,
    default_value: float = 0.0,
    strict: bool = True,
) -> np.ndarray:
    """
    Build a score vector aligned with the exact feature order used in X_train.

    Args:
        feature_names:
            Feature names in the same order as the columns of X_train.
        score_dict:
            Dictionary mapping feature names to feature importance scores.
        default_value:
            Value to assign when strict=False and a feature is missing from score_dict.
        strict:
            If True, raise an error when a feature is not found in score_dict.
            If False, use default_value instead.

    Returns:
        A NumPy array of shape (n_features,) containing the scores in the
        same order as feature_names.
    """
    ordered_scores = []
    missing = []

    for feat in feature_names:
        if feat in score_dict:
            ordered_scores.append(float(score_dict[feat]))
        else:
            if strict:
                missing.append(feat)
            else:
                ordered_scores.append(float(default_value))

    if missing:
        raise KeyError(
            "The following features were not found in score_dict: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    return np.asarray(ordered_scores, dtype=float)
