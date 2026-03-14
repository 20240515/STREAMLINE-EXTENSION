from typing import List
import numpy as np

from .chromosome import Chromosome, init_feature_mask, repair_if_zero
from .fitness import evaluate_fitness_ba_penalized


class Population:
    """
    Represents one generation of chromosomes in the GA.
    Representa uma geração do GA.
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
    ) -> "Population":

        M = X_train.shape[1]
        m_init = max(2, int(np.ceil(m_init_ratio * M)))

        individuals: List[Chromosome] = []

        for _ in range(population_size):

            # --- Feature mask ---
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

            individuals.append(Chromosome(mask, hp))

        return cls(individuals)

    # ---------------------------------------------------
    # Evaluate all individuals
    # ---------------------------------------------------
    def evaluate(self, basemodel, X_train, y_train, lam: float):
        for ind in self.individuals:
            if ind.fitness is None:
                evaluate_fitness_ba_penalized(
                    basemodel,
                    ind,
                    X_train,
                    y_train,
                    lam=lam,
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