import os
import sys
import inspect
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from streamline.optimization.chromosome import Chromosome, init_feature_mask, repair_if_zero
from streamline.optimization.fitness import evaluate_fitness_ba_penalized


def get_model_class_from_module(module):
    """
    Procura no módulo uma classe que pareça ser um modelo STREAMLINE (subclasse de BaseModel).
    Evita teres de adivinhar o nome exato da classe.
    """
    from streamline.modeling.basemodel import BaseModel

    candidates = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseModel) and obj is not BaseModel:
            candidates.append(obj)

    if not candidates:
        raise RuntimeError("Não encontrei subclasses de BaseModel no módulo do modelo.")

    # Heurística: se houver mais do que uma, escolhe a primeira
    return candidates[0]


def main():
    # 1) Dados sintéticos desbalanceados
    X, y = make_classification(
        n_samples=400,
        n_features=60,
        n_informative=12,
        n_redundant=12,
        weights=[0.8, 0.2],
        random_state=42
    )
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 2) Import do modelo real do STREAMLINE (random_forest.py)
    import streamline.models.random_forest as rf_module
    ModelClass = get_model_class_from_module(rf_module)

    # 3) Instanciar BaseModel real
    basemodel = ModelClass(
        cv_folds=3,
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        random_state=42,
        n_jobs=None
    )

    # 4) Cromossoma (init mínimo + repair)
    rng = np.random.default_rng(42)
    M = X_train.shape[1]
    m_init = max(2, int(np.ceil(0.01 * M)))

    mask = init_feature_mask(M, m_init=m_init, rng=rng)
    mask = repair_if_zero(mask, rng=rng)

    # 5) Para este teste, hiperparams vazios (usa defaults do modelo)
    hyperparams = {}
    chrom = Chromosome(feature_mask=mask, hyperparams=hyperparams)

    # 6) Fitness BA penalizada
    lam = 0.05
    fitness = evaluate_fitness_ba_penalized(basemodel, chrom, X_train, y_train, lam=lam)

    print("✅ Integration test (STREAMLINE BaseModel real) OK")
    print(f"Model class = {ModelClass.__name__}")
    print(f"Fitness = {fitness:.4f}")
    print(f"Selected features = {chrom.n_selected()} / {M}")
    print(f"Scoring metric = {basemodel.scoring_metric}")
    print(f"Lambda = {lam}")


if __name__ == "__main__":
    main()