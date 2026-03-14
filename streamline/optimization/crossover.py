# streamline/optimization/crossover.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from .chromosome import Chromosome, repair_chromosome
from .population import Population


@dataclass
class CrossoverConfig:
    """
    Crossover híbrido (FS + HPO).

    mask_method:
      - "uniform": para cada feature, escolhe do pai A ou B com prob 0.5
      - "one_point": corte único na máscara (mais "estrutural")

    hp_method:
      - "uniform": para cada hiperparâmetro, escolhe do pai A ou B
      - "blend": (não implementado aqui) faria sentido só para valores numéricos

    p_crossover:
      - probabilidade de aplicar crossover (caso contrário, clones)

    p_swap_hp:
      - (apenas para hp_method="uniform") probabilidade de herdar cada parâmetro do pai A
        Ex.: 0.5 = metade A / metade B (em média)
    """
    p_crossover: float = 0.9

    mask_method: str = "uniform"
    hp_method: str = "uniform"

    p_swap_hp: float = 0.5  # para uniform hp crossover


# -----------------------------
# Crossover de máscara (features)
# -----------------------------
def _mask_uniform(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover: para cada posição i, troca com prob 0.5.
    """
    a = a.astype(bool, copy=False)
    b = b.astype(bool, copy=False)

    swap = rng.random(size=a.shape[0]) < 0.5
    c1 = a.copy()
    c2 = b.copy()
    c1[swap] = b[swap]
    c2[swap] = a[swap]
    return c1, c2


def _mask_one_point(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-point crossover: escolhe um ponto e troca as "caudas".
    """
    a = a.astype(bool, copy=False)
    b = b.astype(bool, copy=False)

    M = a.shape[0]
    if M <= 1:
        return a.copy(), b.copy()

    cut = int(rng.integers(1, M))  # [1, M-1]
    c1 = np.concatenate([a[:cut], b[cut:]]).astype(bool, copy=False)
    c2 = np.concatenate([b[:cut], a[cut:]]).astype(bool, copy=False)
    return c1, c2


# -----------------------------
# Crossover de hiperparâmetros
# -----------------------------
def _hp_uniform(
    hp_a: Dict[str, Any],
    hp_b: Dict[str, Any],
    rng: np.random.Generator,
    p_from_a: float = 0.5
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Para cada chave (união das chaves dos dois pais), escolhe valor de A ou B.
    """
    keys = sorted(set(hp_a.keys()) | set(hp_b.keys()))
    child1: Dict[str, Any] = {}
    child2: Dict[str, Any] = {}

    for k in keys:
        a_has = k in hp_a
        b_has = k in hp_b

        # se só existir num dos pais, herda esse para os dois (mantém consistência)
        if a_has and not b_has:
            child1[k] = hp_a[k]
            child2[k] = hp_a[k]
            continue
        if b_has and not a_has:
            child1[k] = hp_b[k]
            child2[k] = hp_b[k]
            continue

        # ambos têm
        if rng.random() < p_from_a:
            child1[k] = hp_a[k]
            child2[k] = hp_b[k]
        else:
            child1[k] = hp_b[k]
            child2[k] = hp_a[k]

    return child1, child2


# -----------------------------
# API principal
# -----------------------------
def crossover_pair(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: np.random.Generator,
    cfg: Optional[CrossoverConfig] = None,
) -> Tuple[Chromosome, Chromosome]:
    """
    Faz crossover de 2 pais e devolve 2 filhos.
    Se não aplicar crossover (por probabilidade), devolve clones (copias).
    """
    cfg = cfg or CrossoverConfig()

    # Por defeito: se não houver crossover, fazemos clones (cópias profundas)
    if rng.random() > cfg.p_crossover:
        c1 = parent1.copy()
        c2 = parent2.copy()
        c1.fitness = None
        c2.fitness = None
        return c1, c2

    # --- máscara ---
    if cfg.mask_method == "uniform":
        m1, m2 = _mask_uniform(parent1.feature_mask, parent2.feature_mask, rng)
    elif cfg.mask_method == "one_point":
        m1, m2 = _mask_one_point(parent1.feature_mask, parent2.feature_mask, rng)
    else:
        raise ValueError(f"mask_method desconhecido: {cfg.mask_method}")

    # --- hiperparâmetros ---
    if cfg.hp_method == "uniform":
        hp1, hp2 = _hp_uniform(parent1.hyperparams, parent2.hyperparams, rng, p_from_a=cfg.p_swap_hp)
    else:
        raise ValueError(f"hp_method desconhecido: {cfg.hp_method}")

    # Criar filhos (fitness a None para obrigar reavaliação)
    child1 = Chromosome(feature_mask=m1, hyperparams=hp1, fitness=None)
    child2 = Chromosome(feature_mask=m2, hyperparams=hp2, fitness=None)

    # Repair: garante pelo menos 1 feature, hyperparams != None
    child1 = repair_chromosome(child1, rng)
    child2 = repair_chromosome(child2, rng)

    return child1, child2


def crossover_population(
    parents: List[Chromosome],
    rng: np.random.Generator,
    cfg: Optional[CrossoverConfig] = None,
) -> List[Chromosome]:
    """
    Faz crossover em pares na lista de pais e devolve uma lista de filhos.
    Se o nº de pais for ímpar, o último passa como clone (com fitness=None).
    """
    cfg = cfg or CrossoverConfig()

    offspring: List[Chromosome] = []
    i = 0
    while i + 1 < len(parents):
        p1 = parents[i]
        p2 = parents[i + 1]
        c1, c2 = crossover_pair(p1, p2, rng, cfg)
        offspring.extend([c1, c2])
        i += 2

    # Se ímpar, clona o último
    if i < len(parents):
        last = parents[i].copy()
        last.fitness = None
        offspring.append(repair_chromosome(last, rng))

    return offspring