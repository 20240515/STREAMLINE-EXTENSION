from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import math
import numpy as np


# -------------------------------------------------
# Basic type checking utilities
# -------------------------------------------------

def _is_number(x: Any) -> bool:
    """
    Check if x is a numeric value (int/float or numpy numeric),
    excluding bool (since bool is a subclass of int in Python).
    """
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    """
    Check if x is an integer (including numpy integer types),
    excluding bool.
    """
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


# -------------------------------------------------
# Hyperparameter sampling
# -------------------------------------------------

def sample_one_param(
    spec: Any,
    rng: np.random.Generator,
    *,
    use_log_for_positive_floats: bool = True
) -> Optional[Any]:
    """
    Generate a valid value for a hyperparameter based on its specification.

    Supported specification types (compatible with STREAMLINE parameters.py):

    1) Single value (not list/tuple) -> returned as is
    2) Empty list -> None
    3) List with 1 element -> fixed value
    4) List with 2 numeric elements -> [min, max] range
         - integers -> discrete uniform sampling
         - floats -> uniform or log-uniform (if both positive)
    5) List with multiple elements -> categorical choice
    """

    # Case 1: not a list/tuple -> return directly
    if not isinstance(spec, (list, tuple)):
        return spec

    # Case 2: empty list -> None
    if len(spec) == 0:
        return None

    # Case 3: single element -> fixed value
    if len(spec) == 1:
        return spec[0]

    # Case 4: numeric interval [min, max]
    if len(spec) == 2 and _is_number(spec[0]) and _is_number(spec[1]):
        lo, hi = spec

        if lo > hi:
            lo, hi = hi, lo

        # Integer range
        if _is_int(lo) and _is_int(hi):
            return int(rng.integers(int(lo), int(hi) + 1))

        # Float range
        lo_f, hi_f = float(lo), float(hi)

        # If values are effectively equal
        if math.isclose(lo_f, hi_f):
            return lo_f

        # Log-uniform sampling for strictly positive ranges
        if use_log_for_positive_floats and lo_f > 0 and hi_f > 0:
            u = rng.random()
            return float(math.exp(math.log(lo_f) + u * (math.log(hi_f) - math.log(lo_f))))

        # Standard uniform sampling
        return float(rng.uniform(lo_f, hi_f))

    # Case 5: categorical choice
    return spec[int(rng.integers(0, len(spec)))]


def sample_hyperparams(
    param_grid: Dict[str, Any],
    rng: np.random.Generator,
    *,
    use_log_for_positive_floats: bool = True
) -> Dict[str, Any]:
    """
    Generate a full hyperparameter dictionary based on a STREAMLINE-style
    param_grid.

    Rule: if the sampled value is None, the parameter is not included
    in the final dictionary.
    """
    hp: Dict[str, Any] = {}

    for name, spec in param_grid.items():

        # Ignore special STREAMLINE key
        if name == "expert_knowledge":
            continue

        value = sample_one_param(
            spec,
            rng,
            use_log_for_positive_floats=use_log_for_positive_floats
        )

        if value is not None:
            hp[name] = value

    return hp


# -------------------------------------------------
# Transparency & reproducibility helpers
# -------------------------------------------------

def _convert_to_stable_value(x: Any) -> Any:
    """
    Convert common Python/numpy values into a stable and comparable
    representation suitable for deterministic identification.

    Examples:
      np.int64(5)        -> 5
      np.float64(0.1)    -> 0.1
      [1, 2]             -> (1, 2)
      {"b":2,"a":1}      -> (("a",1),("b",2))  (sorted)
    """
    if x is None:
        return None

    if isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        return float(x)

    if isinstance(x, (list, tuple)):
        return tuple(_convert_to_stable_value(v) for v in x)

    if isinstance(x, dict):
        return tuple(
            (str(k), _convert_to_stable_value(v))
            for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
        )

    # Fallback: stable string representation
    return repr(x)


def hyperparams_key(hp: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """
    Deterministic representation of a hyperparameter dictionary.

    Useful for:
        - fitness caching
        - logging
        - experiment reproducibility

    Example output:
        (("C", 0.1), ("penalty", "l2"), ("solver", "lbfgs"))
    """
    return tuple(
        (str(k), _convert_to_stable_value(v))
        for k, v in sorted(hp.items(), key=lambda kv: str(kv[0]))
    )


# -------------------------------------------------
# Minimal validation / correction layer
# -------------------------------------------------

def validate_and_fix_hyperparams(
    estimator_or_name: Union[Any, str],
    hp: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and minimally correct hyperparameter combinations before
    calling set_params().

    Behavior:
      - If an sklearn estimator is provided: filter out invalid parameter keys.
      - Apply minimal compatibility rules for:
            * LogisticRegression
            * SVC

    This prevents common configuration crashes during GA evaluation.
    """
    hp = dict(hp)

    est = None
    name = None

    if isinstance(estimator_or_name, str):
        name = estimator_or_name
    else:
        est = estimator_or_name
        name = est.__class__.__name__

        # Filter out invalid parameter keys
        try:
            valid_keys = set(est.get_params(deep=True).keys())
            hp = {k: v for k, v in hp.items() if k in valid_keys}
        except Exception:
            pass

    # ---- LogisticRegression rules ----
    if name == "LogisticRegression":

        solver = hp.get("solver", None)
        penalty = hp.get("penalty", None)

        allowed = {
            "lbfgs": {"l2", "none", None},
            "newton-cg": {"l2", "none", None},
            "sag": {"l2", "none", None},
            "saga": {"l1", "l2", "elasticnet", "none", None},
            "liblinear": {"l1", "l2"},
        }

        # If penalty is incompatible with solver -> force safe default
        if solver in allowed and penalty is not None and penalty not in allowed[solver]:
            hp["penalty"] = "l2"

        # elasticnet requires saga + l1_ratio
        if hp.get("penalty") == "elasticnet":
            hp["solver"] = "saga"
            if "l1_ratio" not in hp or hp["l1_ratio"] is None:
                hp["l1_ratio"] = 0.5
        else:
            hp.pop("l1_ratio", None)

        # dual is only valid for liblinear + l2
        if "dual" in hp:
            if hp.get("solver") != "liblinear" or hp.get("penalty") != "l2":
                hp["dual"] = False

    # ---- SVC rules ----
    if name == "SVC":

        kernel = hp.get("kernel", None)

        # degree only relevant for polynomial kernel
        if kernel != "poly":
            hp.pop("degree", None)

        # gamma only relevant for rbf/poly/sigmoid
        if kernel not in {"rbf", "poly", "sigmoid"}:
            hp.pop("gamma", None)

    return hp