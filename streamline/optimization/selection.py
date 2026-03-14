# streamline/optimization/selection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .chromosome import Chromosome
from .population import Population


@dataclass
class SelectionConfig:
    """
    Seleção single-objective (fitness maior é melhor).

    method:
      - "tournament"  : recomendado (robusto a ruído de CV)
      - "rank"        : recomendado (menos sensível a outliers)
      - "truncation"  : baseline agressivo
      - "random"      : baseline

    Parâmetros:
      - tournament_k controla pressão seletiva (2 = suave, 3 = equilíbrio, 5 = agressivo)
      - rank_eta em [1, 2] controla pressão seletiva no rank selection
      - truncation_top_frac define percentagem do topo onde é permitido escolher pais
    """
    method: str = "tournament"

    tournament_k: int = 3
    rank_eta: float = 1.7
    truncation_top_frac: float = 0.5

    # coerente com evaluate_fitness_ba_penalized(failure_fitness=-1.0)
    invalid_fitness_value: float = -1.0


def _fitness_array(pop: Population, invalid_value: float) -> np.ndarray:
    """
    Converte fitness da população num array float.
    Se fitness for None/NaN/inf, usa invalid_value.
    """
    individuals = pop.individuals
    f = np.empty(len(individuals), dtype=float)

    for i, ind in enumerate(individuals):
        val = ind.fitness
        if val is None:
            f[i] = invalid_value
        else:
            v = float(val)
            f[i] = v if np.isfinite(v) else invalid_value

    return f


# -----------------------------
# Métodos de seleção
# -----------------------------
def selection_random(pop: Population, n_select: int, rng: np.random.Generator) -> List[Chromosome]:
    individuals = pop.individuals
    idx = rng.integers(0, len(individuals), size=n_select)
    return [individuals[int(i)] for i in idx]


def selection_tournament(
    pop: Population,
    n_select: int,
    rng: np.random.Generator,
    k: int,
    invalid_fitness_value: float,
) -> List[Chromosome]:
    """
    Tournament selection:
      - escolhe k candidatos aleatórios
      - devolve o melhor (maior fitness)
    Repetir até obter n_select pais.
    """
    individuals = pop.individuals
    if len(individuals) == 0:
        return []

    scores = _fitness_array(pop, invalid_fitness_value)
    N = len(individuals)
    k = max(2, int(k))

    selected: List[Chromosome] = []
    for _ in range(n_select):
        cand_idx = rng.integers(0, N, size=k)
        best_idx = int(cand_idx[np.argmax(scores[cand_idx])])
        selected.append(individuals[best_idx])

    return selected


def selection_rank_linear(
    pop: Population,
    n_select: int,
    rng: np.random.Generator,
    eta: float,
    invalid_fitness_value: float,
) -> List[Chromosome]:
    """
    Linear rank selection (Baker):
    - ordena por fitness
    - atribui probabilidade baseada no rank, não no valor absoluto

    eta em [1,2]:
      - 1 -> quase uniforme (pouca pressão seletiva)
      - 2 -> muito agressivo (muita pressão seletiva)
    """
    individuals = pop.individuals
    if len(individuals) == 0:
        return []

    scores = _fitness_array(pop, invalid_fitness_value)
    N = len(individuals)

    eta = float(eta)
    eta = min(2.0, max(1.0, eta))

    # melhores primeiro
    order = np.argsort(-scores)

    # rank: 0 melhor ... N-1 pior
    ranks = np.empty(N, dtype=int)
    ranks[order] = np.arange(N)

    # i: 0 pior ... N-1 melhor (para fórmula clássica)
    i = (N - 1) - ranks

    denom = (N * (N - 1)) if N > 1 else 1
    p = (2.0 - eta) / N + (2.0 * i * (eta - 1.0)) / denom
    p = p / float(np.sum(p))

    idx = rng.choice(N, size=n_select, replace=True, p=p)
    return [individuals[int(j)] for j in idx]


def selection_truncation(
    pop: Population,
    n_select: int,
    rng: np.random.Generator,
    top_frac: float,
    invalid_fitness_value: float,
) -> List[Chromosome]:
    """
    Truncation selection:
    - só permite escolher pais do top X% da população (por fitness)
    """
    individuals = pop.individuals
    if len(individuals) == 0:
        return []

    scores = _fitness_array(pop, invalid_fitness_value)

    top_frac = float(top_frac)
    top_frac = min(1.0, max(0.05, top_frac))

    N = len(individuals)
    k = max(1, int(round(N * top_frac)))

    top_idx = np.argsort(-scores)[:k]
    chosen = rng.choice(top_idx, size=n_select, replace=True)
    return [individuals[int(j)] for j in chosen]


# -----------------------------
# Interface única (para o teu GA)
# -----------------------------
def select_parents(
    pop: Population,
    n_select: int,
    rng: np.random.Generator,
    cfg: Optional[SelectionConfig] = None,
) -> List[Chromosome]:
    """
    Seleciona n_select pais da população, de acordo com cfg.method.
    """
    cfg = cfg or SelectionConfig()
    method = cfg.method.lower().strip()

    if method == "tournament":
        return selection_tournament(
            pop,
            n_select,
            rng,
            k=cfg.tournament_k,
            invalid_fitness_value=cfg.invalid_fitness_value,
        )

    if method == "rank":
        return selection_rank_linear(
            pop,
            n_select,
            rng,
            eta=cfg.rank_eta,
            invalid_fitness_value=cfg.invalid_fitness_value,
        )

    if method == "truncation":
        return selection_truncation(
            pop,
            n_select,
            rng,
            top_frac=cfg.truncation_top_frac,
            invalid_fitness_value=cfg.invalid_fitness_value,
        )

    if method == "random":
        return selection_random(pop, n_select, rng)

    raise ValueError(f"Selection method desconhecido: {cfg.method}")