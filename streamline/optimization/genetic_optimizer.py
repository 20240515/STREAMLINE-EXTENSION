from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from streamline.utils import fi_scores

from .chromosome import Chromosome
from .population import Population
from .fitness import FitnessCache
from .selection import SelectionConfig, select_parents
from .crossover import CrossoverConfig, crossover_population
from .mutation import MutationConfig, mutate_population


@dataclass
class GeneticOptimizerConfig:
    """
    Configuration for one GA run.
    """
    population_size: int = 30
    n_generations: int = 20
    m_init_ratio: float = 0.2

    elitism: int = 2
    lam: float = 0.05

    use_importance_bias: bool = True
    return_history: bool = True

    # Early stopping
    early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-3
    early_stop_min_generations: int = 5

    selection_cfg: SelectionConfig = field(default_factory=SelectionConfig)
    crossover_cfg: CrossoverConfig = field(default_factory=CrossoverConfig)
    mutation_cfg: MutationConfig = field(default_factory=MutationConfig)


class GeneticOptimizer:
    """
    Orchestrates the GA for a single base model.
    """

    def __init__(self, cfg: Optional[GeneticOptimizerConfig] = None):


        if isinstance(cfg, dict):


            self.cfg = GeneticOptimizerConfig(
                population_size=cfg.get("population_size", 30),
                n_generations=cfg.get("n_generations", 20),
                elitism=cfg.get("elitism", 2),
                early_stop=cfg.get("early_stop", False),
                early_stop_patience=cfg.get("early_stop_patience", 5),
                early_stop_min_delta=cfg.get("early_stop_min_delta", 1e-3),
                early_stop_min_generations=cfg.get("early_stop_min_generations", 5),
                selection_cfg=SelectionConfig(**cfg.get("selection", {})),
                crossover_cfg=CrossoverConfig(**cfg.get("crossover", {})),
                mutation_cfg=MutationConfig(**cfg.get("mutation", {})),
            )
        elif cfg is None:
            self.cfg = GeneticOptimizerConfig()
        else:
            self.cfg = cfg


    def _make_history_row(
        self,
        generation: int,
        population: Population,
        cache: Optional[FitnessCache] = None,
    ) -> Dict[str, Any]:
        fitness_vals = [
            float(ind.fitness)
            for ind in population.individuals
            if ind.fitness is not None and np.isfinite(ind.fitness)
        ]

        n_selected = [ind.n_selected() for ind in population.individuals]

        row = {
            "generation": int(generation),
            "population_size": int(len(population)),
            "best_fitness": None,
            "mean_fitness": None,
            "worst_fitness": None,
            "best_n_selected": None,
            "mean_n_selected": None,
            "cache_hits": None,
            "cache_misses": None,
            "early_stop_triggered": False,
        }

        if fitness_vals:
            row["best_fitness"] = float(np.max(fitness_vals))
            row["mean_fitness"] = float(np.mean(fitness_vals))
            row["worst_fitness"] = float(np.min(fitness_vals))

        if n_selected:
            row["mean_n_selected"] = float(np.mean(n_selected))
            best = population.best()
            if best is not None:
                row["best_n_selected"] = int(best.n_selected())

        if cache is not None:
            row["cache_hits"] = int(cache.hits)
            row["cache_misses"] = int(cache.misses)

        return row

    def _copy_elites(self, population: Population) -> List[Chromosome]:
        n_elite = max(0, min(int(self.cfg.elitism), len(population)))
        return [population.individuals[i].copy() for i in range(n_elite)]

    def _fill_offspring_if_needed(
        self,
        basemodel,
        X_train: np.ndarray,
        offspring: List[Chromosome],
        rng: np.random.Generator,
        fi_scores: Optional[np.ndarray],
        n_needed: int,
    ) -> List[Chromosome]:
        while len(offspring) < n_needed:
            extra_pop = Population.random_init(
                basemodel=basemodel,
                X_train=X_train,
                population_size=1,
                m_init_ratio=float(self.cfg.m_init_ratio),
                rng=rng,
                fi_scores=fi_scores,
                use_importance_bias=bool(self.cfg.use_importance_bias),
            )
            offspring.extend(extra_pop.individuals)

        return offspring[:n_needed]

    def optimize(
        self,
        basemodel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        rng: np.random.Generator,
        fi_scores: Optional[np.ndarray] = None,
        cache_seed: Optional[int] = None,
    ):
        history: List[Dict[str, Any]] = []
        fitness_cache = FitnessCache()


        print("\n[DEBUG before Population.random_init]")
        print("use_importance_bias =", self.cfg.use_importance_bias)
        print("fi_scores is None =", fi_scores is None)
        print("n fi_scores =", None if fi_scores is None else len(fi_scores))

        population = Population.random_init(
            basemodel=basemodel,
            X_train=X_train,
            population_size=int(self.cfg.population_size),
            m_init_ratio=float(self.cfg.m_init_ratio),
            rng=rng,
            fi_scores=fi_scores,
            use_importance_bias=bool(self.cfg.use_importance_bias),
        )

        print(f"Initial population of size {len(population)} created.")
        print(f"type(population) = {type(population)}")


        population.evaluate(
            basemodel=basemodel,
            X_train=X_train,
            y_train=y_train,
            lam=float(self.cfg.lam),
            cache=fitness_cache,
            cache_seed=cache_seed,
        )
        population.sort()

        if self.cfg.return_history:
            history.append(
                self._make_history_row(
                    generation=0,
                    population=population,
                    cache=fitness_cache,
                )
            )

        best_so_far = None
        no_improve_count = 0

        current_best = population.best()
        if current_best is not None and current_best.fitness is not None:
            best_so_far = float(current_best.fitness)

        for gen in range(1, int(self.cfg.n_generations) + 1):
            population.sort()

            elites = self._copy_elites(population)
            n_offspring_needed = int(self.cfg.population_size) - len(elites)

            if n_offspring_needed <= 0:
                population = Population(elites[: int(self.cfg.population_size)])
                population.sort()

                row = None
                if self.cfg.return_history:
                    row = self._make_history_row(
                        generation=gen,
                        population=population,
                        cache=fitness_cache,
                    )
                    history.append(row)

                # early stopping check
                current_best = population.best()
                if current_best is not None and current_best.fitness is not None:
                    current_best_fit = float(current_best.fitness)

                    if best_so_far is None or current_best_fit > best_so_far + float(self.cfg.early_stop_min_delta):
                        best_so_far = current_best_fit
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    if (
                        self.cfg.early_stop
                        and gen >= int(self.cfg.early_stop_min_generations)
                        and no_improve_count >= int(self.cfg.early_stop_patience)
                    ):
                        if row is not None:
                            row["early_stop_triggered"] = True
                        break

                continue

            parents = select_parents(
                pop=population,
                n_select=n_offspring_needed,
                rng=rng,
                cfg=self.cfg.selection_cfg,
            )

            offspring = crossover_population(
                parents=parents,
                rng=rng,
                cfg=self.cfg.crossover_cfg,
            )

            offspring = mutate_population(
                individuals=offspring,
                basemodel=basemodel,
                rng=rng,
                cfg=self.cfg.mutation_cfg,
            )

            offspring = self._fill_offspring_if_needed(
                basemodel=basemodel,
                X_train=X_train,
                offspring=offspring,
                rng=rng,
                fi_scores=fi_scores,
                n_needed=n_offspring_needed,
            )

            next_population = Population(elites + offspring[:n_offspring_needed])

            next_population.evaluate(
                basemodel=basemodel,
                X_train=X_train,
                y_train=y_train,
                lam=float(self.cfg.lam),
                cache=fitness_cache,
                cache_seed=cache_seed,
            )
            next_population.sort()
            population = next_population

            row = None
            if self.cfg.return_history:
                row = self._make_history_row(
                    generation=gen,
                    population=population,
                    cache=fitness_cache,
                )
                history.append(row)

            current_best = population.best()
            if current_best is not None and current_best.fitness is not None:
                current_best_fit = float(current_best.fitness)

                if best_so_far is None or current_best_fit > best_so_far + float(self.cfg.early_stop_min_delta):
                    best_so_far = current_best_fit
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if (
                    self.cfg.early_stop
                    and gen >= int(self.cfg.early_stop_min_generations)
                    and no_improve_count >= int(self.cfg.early_stop_patience)
                ):
                    if row is not None:
                        row["early_stop_triggered"] = True
                    break

        population.sort()
        best = population.best()

        if self.cfg.return_history:
            return best, history
        return best