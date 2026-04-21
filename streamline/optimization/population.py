from typing import List, Optional
import numpy as np

from .chromosome import Chromosome, init_feature_mask, repair_if_zero
from .fitness import evaluate_fitness_ba_penalized


def _normalize_importance_scores(fi_scores: Optional[np.ndarray], M: int) -> Optional[np.ndarray]:
    """
    Convert raw feature-importance scores into a valid probability vector.

    Rules:
      - If fi_scores is None, return None.
      - Length must match the number of features.
      - NaN / inf / negative values are made safe.
      - If all scores become zero, return None.

    Returns:
        A probability vector of shape (M,) or None if normalization is not possible.
    """
    if fi_scores is None:
        return None

    scores = np.asarray(fi_scores, dtype=float)

    if scores.shape[0] != M:
        raise ValueError(
            f"fi_scores length ({scores.shape[0]}) does not match number of features ({M})."
        )

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)

    total = scores.sum()
    if total <= 0.0:
        return None

    return scores / total


def _init_feature_mask_biased(
    M: int,
    m_init: int,
    rng: np.random.Generator,
    fi_scores: Optional[np.ndarray],
) -> np.ndarray:
    """
    Initialize a feature mask using feature-importance-guided sampling.

    Behavior:
      - If fi_scores is valid, sample features without replacement using
        importance-derived probabilities.
      - If fi_scores is None or unusable, fall back to uniform initialization.
    """
    probs = _normalize_importance_scores(fi_scores, M)

    if probs is None:
        return init_feature_mask(M, m_init=m_init, rng=rng)

    m_init = max(1, min(m_init, M))

    mask = np.zeros(M, dtype=bool)
    idx = rng.choice(M, size=m_init, replace=False, p=probs)
    mask[idx] = True
    return mask


class Population:
    """
    Represents one generation of chromosomes in the GA.
    """

    def __init__(self, individuals: List[Chromosome]):
        self.individuals = individuals

    # ---------------------------------------------------
    # Create initial random population
    # ---------------------------------------------------
    @classmethod
    def random_init(
        cls,
        basemodel,
        X_train: np.ndarray,
        population_size: int,
        m_init_ratio: float,
        rng: np.random.Generator,
        fi_scores: Optional[np.ndarray] = None,
        use_importance_bias: bool = False,
    ) -> "Population":

        M = X_train.shape[1]
        m_init = max(2, int(np.ceil(m_init_ratio * M)))

        individuals: List[Chromosome] = []

        for _ in range(population_size):

            # --- Feature mask ---
            if use_importance_bias:
                print("[DEBUG Population.random_init] Using importance bias for feature mask initialization.")
                mask = _init_feature_mask_biased(
                    M=M,
                    m_init=m_init,
                    rng=rng,
                    fi_scores=fi_scores,
                )
            else:
                mask = init_feature_mask(M, m_init=m_init, rng=rng)

            mask = repair_if_zero(mask, rng=rng)

            # --- Hyperparameters ---
            hp = {}
            for k, v in basemodel.param_grid.items():
                if k == "expert_knowledge":
                    continue

                if isinstance(v, (list, tuple)) and len(v) > 0:
                    hp[k] = v[int(rng.integers(0, len(v)))]
                else:
                    hp[k] = v

            chrom = Chromosome(mask, hp)

            # Optional metadata for transparency/debugging
            chrom.meta["init_strategy"] = (
                "importance_biased" if use_importance_bias and fi_scores is not None else "uniform"
            )
            chrom.meta["n_selected_init"] = chrom.n_selected()

            individuals.append(chrom)

        return cls(individuals)

    # ---------------------------------------------------
    # Evaluate all individuals
    # ---------------------------------------------------
    def evaluate(
        self,
        basemodel,
        X_train,
        y_train,
        lam: float,
        cache=None,
        cache_seed=None,
    ):
        for ind in self.individuals:
            if ind.fitness is None:
                evaluate_fitness_ba_penalized(
                    basemodel,
                    ind,
                    X_train,
                    y_train,
                    lam=lam,
                    cache=cache,
                    cache_seed=cache_seed,
                )

    # ---------------------------------------------------
    # Sort population by fitness (descending)
    # ---------------------------------------------------
    def sort(self):
        self.individuals.sort(key=lambda c: c.fitness, reverse=True)

    # ---------------------------------------------------
    # Return best individual
    # ---------------------------------------------------
    def best(self) -> Chromosome:
        return self.individuals[0]

    # ---------------------------------------------------
    # Utility methods
    # ---------------------------------------------------
    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def __iter__(self):
        return iter(self.individuals)