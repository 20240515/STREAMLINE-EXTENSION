# streamline/optimization/mutation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .chromosome import Chromosome, repair_chromosome


@dataclass
class MutationConfig:
    """
    Mutação híbrida (FS + HPO).

    FS (feature mask):
      - p_flip: probabilidade de flip por feature (bit flip).
        Ex.: p_flip=1/M dá ~1 flip por indivíduo em média.
      - max_flips: limite opcional para não virar a máscara toda do avesso.

    HPO (hyperparams):
      - p_hp_mut: probabilidade de mutar hiperparâmetros (por indivíduo).
      - hp_mut_frac: fração de hiperparâmetros a mutar quando decide mutar.
        Ex.: 0.3 = muta ~30% das chaves.
      - hp_mut_min: mínimo de hiperparâmetros a mutar quando decide mutar.

    Nota: usamos o basemodel.param_grid como “fonte” para resampling.
    """
    # --- FS ---
    p_flip: Optional[float] = None   # se None, usamos 1/M
    max_flips: Optional[int] = None  # se None, sem limite

    # --- HPO ---
    p_hp_mut: float = 0.3
    hp_mut_frac: float = 0.3
    hp_mut_min: int = 1


# -----------------------------
# Helpers
# -----------------------------
def _sample_from_grid(rng: np.random.Generator, values: Any) -> Any:
    """
    Dado um item do param_grid, escolhe um valor.
    - Se for lista/tuplo: escolhe um elemento aleatório
    - Caso contrário: devolve o próprio valor (constante)
    """
    if isinstance(values, (list, tuple)) and len(values) > 0:
        return values[int(rng.integers(0, len(values)))]
    return values


def _mutate_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    p_flip: float,
    max_flips: Optional[int],
) -> np.ndarray:
    """
    Bit-flip mutation: cada posição tem probabilidade p_flip de inverter.
    """
    M = mask.shape[0]
    new_mask = mask.astype(bool, copy=True)

    flip = rng.random(M) < p_flip
    if max_flips is not None:
        # se houver demasiados flips, escolhe apenas max_flips
        idx = np.where(flip)[0]
        if len(idx) > max_flips:
            keep = rng.choice(idx, size=max_flips, replace=False)
            flip = np.zeros(M, dtype=bool)
            flip[keep] = True

    new_mask[flip] = ~new_mask[flip]
    return new_mask


def _mutate_hyperparams(
    hp: Dict[str, Any],
    basemodel,
    rng: np.random.Generator,
    hp_mut_frac: float,
    hp_mut_min: int,
) -> Dict[str, Any]:
    """
    Mutação por resampling de algumas chaves do param_grid.
    """
    new_hp = dict(hp)

    # param_grid do STREAMLINE inclui possivelmente "expert_knowledge"
    grid = getattr(basemodel, "param_grid", {}) or {}
    keys = [k for k in grid.keys() if k != "expert_knowledge"]

    if len(keys) == 0:
        return new_hp

    # nº de chaves a mutar
    k = int(round(hp_mut_frac * len(keys)))
    k = max(int(hp_mut_min), k)
    k = min(k, len(keys))

    chosen = rng.choice(keys, size=k, replace=False)

    for key in chosen:
        new_hp[key] = _sample_from_grid(rng, grid[key])

    return new_hp


# -----------------------------
# API principal
# -----------------------------
def mutate_individual(
    ind: Chromosome,
    basemodel,
    rng: np.random.Generator,
    cfg: Optional[MutationConfig] = None,
) -> Chromosome:
    """
    Aplica mutação a um indivíduo e devolve um novo Chromosome (sem side effects).
    - máscara: bit flips
    - hiperparams: resampling parcial (com prob p_hp_mut)
    - fitness: fica None para obrigar reavaliação
    """
    cfg = cfg or MutationConfig()

    child = ind.copy()
    child.fitness = None  # importante: depois tens de reavaliar

    # --- FS mutation ---
    M = int(child.feature_mask.shape[0])
    p_flip = cfg.p_flip if cfg.p_flip is not None else (1.0 / max(1, M))
    p_flip = float(p_flip)

    if p_flip > 0.0:
        child.feature_mask = _mutate_mask(child.feature_mask, rng, p_flip, cfg.max_flips)

    # --- HPO mutation ---
    if rng.random() < float(cfg.p_hp_mut):
        child.hyperparams = _mutate_hyperparams(
            child.hyperparams,
            basemodel,
            rng,
            hp_mut_frac=float(cfg.hp_mut_frac),
            hp_mut_min=int(cfg.hp_mut_min),
        )

    # --- Repair ---
    child = repair_chromosome(child, rng)

    return child


def mutate_population(
    individuals: List[Chromosome],
    basemodel,
    rng: np.random.Generator,
    cfg: Optional[MutationConfig] = None,
) -> List[Chromosome]:
    """
    Aplica mutação a uma lista de indivíduos (ex.: offspring).
    """
    cfg = cfg or MutationConfig()
    return [mutate_individual(ind, basemodel, rng, cfg) for ind in individuals]