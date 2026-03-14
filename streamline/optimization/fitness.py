from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import cross_val_score

from .hyperparameters import validate_and_fix_hyperparams


# -------------------------------------------------
# Minimal fitness cache (optional)
# -------------------------------------------------

@dataclass
class FitnessCache:
    """
    Simple in-memory cache to avoid re-evaluating identical individuals.

    This is intentionally minimal:
      - dictionary-based
      - no eviction policy
      - per GA run
    """
    values: Dict[Tuple[Any, ...], float] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: Tuple[Any, ...]) -> Optional[float]:
        if key in self.values:
            self.hits += 1
            return self.values[key]
        self.misses += 1
        return None

    def set(self, key: Tuple[Any, ...], value: float) -> None:
        self.values[key] = float(value)


def make_fitness_cache_key(
    basemodel: Any,
    chromosome: Any,
    X_shape: Tuple[int, int],
    lam: float,
    seed: Optional[int] = None
) -> Tuple[Any, ...]:
    """
    Deterministic cache key including context.

    We intentionally include:
      - number of features (M)
      - lam
      - seed (optional)
      - estimator name
      - scoring metric
      - CV representation
      - genotype signature (mask + hyperparams)

    This avoids collisions across:
      - different datasets
      - different models
      - different scoring metrics
      - different GA runs
    """
    M = int(X_shape[1])
    mask_bytes, hp_key = chromosome.genotype_signature()

    try:
        est_name = basemodel.model.__class__.__name__
    except Exception:
        est_name = str(type(getattr(basemodel, "model", None)).__name__)

    scoring = str(getattr(basemodel, "scoring_metric", "balanced_accuracy"))
    cv_repr = repr(getattr(basemodel, "cv", None))

    return (M, float(lam), seed, est_name, scoring, cv_repr, mask_bytes, hp_key)


# -------------------------------------------------
# Fitness evaluation
# -------------------------------------------------

def evaluate_fitness_ba_penalized(
    basemodel,
    chromosome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    lam: float = 0.05,
    failure_fitness: float = -1.0,
    *,
    cache: Optional[FitnessCache] = None,
    cache_seed: Optional[int] = None
) -> float:
    """
    Fitness of a chromosome using STREAMLINE-like logic:

      - Stratified CV (basemodel.cv)
      - scoring = basemodel.scoring_metric (default: balanced_accuracy)
      - estimator = deepcopy(basemodel.model) with set_params(hyperparams)
      - fitness = mean_CV_score - lam*(m/M)

    Minimal additions for robustness:
      - hyperparameter key filtering + validation/fix
      - try/except around cross_val_score to avoid GA crashes
      - optional fitness cache (avoid re-evaluating clones)
      - store fail_reason/meta if those fields exist in the chromosome
    """
    start_time = time.perf_counter()

    M = int(X_train.shape[1])
    selected = chromosome.selected_indices()
    m = int(len(selected))

    # Extra safety (in case repair was bypassed)
    if m == 0:
        chromosome.fitness = failure_fitness
        if hasattr(chromosome, "fail_reason"):
            chromosome.fail_reason = "zero_features_selected"
        if hasattr(chromosome, "meta"):
            chromosome.meta = {
                "M": M,
                "m": m,
                "complexity_ratio": 0.0,
                "lam": float(lam),
                "scoring_metric": str(getattr(basemodel, "scoring_metric", "balanced_accuracy")),
                "elapsed_sec": float(time.perf_counter() - start_time),
                "cache_hit": False,
            }
        return chromosome.fitness

    complexity_ratio = m / M

    # -------------------------------------------------
    # Cache lookup (before doing CV)
    # -------------------------------------------------

    cache_hit = False
    if cache is not None:
        key = make_fitness_cache_key(
            basemodel,
            chromosome,
            X_train.shape,
            lam,
            cache_seed
        )
        cached_value = cache.get(key)
        if cached_value is not None:
            cache_hit = True
            chromosome.fitness = float(cached_value)

            if hasattr(chromosome, "fail_reason"):
                chromosome.fail_reason = None
            if hasattr(chromosome, "meta"):
                chromosome.meta = {
                    "M": M,
                    "m": m,
                    "complexity_ratio": float(complexity_ratio),
                    "lam": float(lam),
                    "scoring_metric": str(getattr(basemodel, "scoring_metric", "balanced_accuracy")),
                    "elapsed_sec": float(time.perf_counter() - start_time),
                    "cache_hit": True,
                }
            return chromosome.fitness

    X_sel = X_train[:, selected]

    # -------------------------------------------------
    # Build model with chromosome hyperparameters
    # -------------------------------------------------

    try:
        base_est = copy.deepcopy(basemodel.model)

        # Filter unsupported hyperparameter keys (safety guard)
        try:
            valid_keys = set(base_est.get_params(deep=True).keys())
            hp = {k: v for k, v in chromosome.hyperparams.items() if k in valid_keys}
        except Exception:
            hp = dict(chromosome.hyperparams)

        # Validate/fix common incompatible combinations (LR/SVC, etc.)
        hp = validate_and_fix_hyperparams(base_est, hp)

        model = base_est.set_params(**hp)

    except Exception as e:
        chromosome.fitness = failure_fitness
        if hasattr(chromosome, "fail_reason"):
            chromosome.fail_reason = f"model_build_error: {type(e).__name__}: {e}"
        if hasattr(chromosome, "meta"):
            chromosome.meta = {
                "M": M,
                "m": m,
                "complexity_ratio": float(complexity_ratio),
                "lam": float(lam),
                "scoring_metric": str(getattr(basemodel, "scoring_metric", "balanced_accuracy")),
                "elapsed_sec": float(time.perf_counter() - start_time),
                "cache_hit": cache_hit,
            }
        return chromosome.fitness

    try:
        scores = cross_val_score(
            model,
            X_sel,
            y_train,
            scoring=basemodel.scoring_metric,
            cv=basemodel.cv,
            n_jobs=basemodel.n_jobs
        )

        mean_score = float(scores.mean())
        std_score = float(scores.std(ddof=0))

        fitness = float(mean_score - lam * complexity_ratio)
        chromosome.fitness = fitness

        chromosome.hyperparams = hp  # keep validated version

        if hasattr(chromosome, "fail_reason"):
            chromosome.fail_reason = None
        if hasattr(chromosome, "meta"):
            chromosome.meta = {
                "mean_score": mean_score,
                "std_score": std_score,
                "M": M,
                "m": m,
                "complexity_ratio": float(complexity_ratio),
                "lam": float(lam),
                "scoring_metric": str(basemodel.scoring_metric),
                "elapsed_sec": float(time.perf_counter() - start_time),
                "cache_hit": cache_hit,
            }

        # -------------------------------------------------
        # Cache store
        # -------------------------------------------------

        if cache is not None:
            cache.set(key, chromosome.fitness)

        return chromosome.fitness

    except Exception as e:
        chromosome.fitness = failure_fitness
        if hasattr(chromosome, "fail_reason"):
            chromosome.fail_reason = f"cv_error: {type(e).__name__}: {e}"
        if hasattr(chromosome, "meta"):
            chromosome.meta = {
                "M": M,
                "m": m,
                "complexity_ratio": float(complexity_ratio),
                "lam": float(lam),
                "scoring_metric": str(getattr(basemodel, "scoring_metric", "balanced_accuracy")),
                "elapsed_sec": float(time.perf_counter() - start_time),
                "cache_hit": cache_hit,
            }
        return chromosome.fitness