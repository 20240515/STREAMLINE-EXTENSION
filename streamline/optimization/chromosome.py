from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import copy  # NEW: needed for deep copies of nested structures
import numpy as np

from .hyperparameters import hyperparams_key


@dataclass
class Chromosome:
    """
    Hybrid chromosome for:
      - Feature Selection (FS): feature_mask (binary vector)
      - Hyperparameter Optimization (HPO): hyperparams (dict)
      - Fitness: fitness (float)

    Notes:
      - The chromosome remains model-agnostic (the model is provided externally to the fitness function).
      - meta and fail_reason support transparency, debugging, and reproducibility.
    """
    feature_mask: np.ndarray                 # shape (M,), dtype bool or {0,1}
    hyperparams: Dict[str, Any]              # e.g. {"max_depth": 7, "n_estimators": 200}
    fitness: Optional[float] = None

    # Transparency / debugging fields
    meta: Dict[str, Any] = field(default_factory=dict)
    fail_reason: Optional[str] = None

    def selected_indices(self) -> np.ndarray:
        """
        Return indices of selected features.
        """
        return np.where(self.feature_mask.astype(bool))[0]

    def n_selected(self) -> int:
        """
        Return the number of selected features.
        """
        return int(self.feature_mask.astype(bool).sum())

    def copy(self) -> "Chromosome":
        """
        Deep copy of the chromosome.

        NOTE:
          - dict.copy() is a shallow copy (nested objects can still be shared).
          - In a GA, this can cause subtle bugs (e.g., meta/hyperparams being mutated across individuals).
          - deepcopy keeps things robust without adding complexity to the rest of the code.
        """
        return Chromosome(
            feature_mask=self.feature_mask.copy(),
            hyperparams=copy.deepcopy(self.hyperparams),  # CHANGED: true deep copy
            fitness=self.fitness,
            meta=copy.deepcopy(self.meta),                # CHANGED: true deep copy
            fail_reason=self.fail_reason
        )

    def genotype_signature(self) -> Tuple[bytes, Tuple[Tuple[str, Any], ...]]:
        """
        Deterministic signature of the genotype (features + hyperparameters).

        This is useful for:
          - fitness caching (avoid re-evaluating identical individuals)
          - duplicate detection
          - logging / experiment reproducibility

        Returns:
          (mask_bytes, hyperparams_key)
        """
        mask_bytes = np.packbits(self.feature_mask.astype(np.uint8)).tobytes()
        hp_key = hyperparams_key(self.hyperparams)
        return (mask_bytes, hp_key)


def init_feature_mask(M: int, m_init: int, rng: np.random.Generator) -> np.ndarray:
    """
    Initialize a mask with at least m_init features set to True.
    """
    m_init = max(1, min(m_init, M))
    mask = np.zeros(M, dtype=bool)
    idx = rng.choice(M, size=m_init, replace=False)
    mask[idx] = True
    return mask


def repair_if_zero(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Repair function: if all features are off, turn on one random feature.
    """
    if mask.astype(bool).sum() == 0:
        j = int(rng.integers(0, len(mask)))
        mask[j] = True
    return mask


def repair_chromosome(chrom: Chromosome, rng: np.random.Generator) -> Chromosome:
    """
    Minimal and consistent repair:
      - ensure at least one feature is selected
      - ensure hyperparams is always a dict (not None)

    NOTE:
      - This function used to mutate the chromosome in-place.
      - In a GA with elitism / reuse of individuals, in-place repair can create side effects.
      - Returning a repaired copy keeps behavior predictable and avoids hard-to-debug issues.
    """
    chrom = chrom.copy()  # NEW: avoid mutating the input object in-place
    chrom.feature_mask = repair_if_zero(chrom.feature_mask, rng)
    if chrom.hyperparams is None:
        chrom.hyperparams = {}
    return chrom