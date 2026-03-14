import os
import sys
import numpy as np

# Garante imports a partir da raiz do projeto (funciona mesmo sem instalar como package)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Importa o teu código GA (ajusta o path conforme a pasta onde o colocaste)
# Exemplo: streamline/optimization/ga/chromosome.py e fitness.py
from streamline.optimization.chromosome import Chromosome, init_feature_mask, repair_if_zero
from streamline.optimization.fitness import evaluate_fitness_ba_penalized


class BaseModelMock:
    """
    Mock mínimo do BaseModel do STREAMLINE:
    só com os atributos usados na tua fitness.
    """
    def __init__(self, random_state=42, cv_folds=3, n_jobs=None):
        self.model = RandomForestClassifier(random_state=random_state)
        self.scoring_metric = "balanced_accuracy"
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.n_jobs = n_jobs


def main():
    # 1) Dados sintéticos desbalanceados (realista para saúde)
    X, y = make_classification(
        n_samples=400,
        n_features=60,
        n_informative=12,
        n_redundant=12,
        weights=[0.8, 0.2],
        random_state=42
    )
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 2) BaseModel “mock” (mais tarde trocas pelo BaseModel real do STREAMLINE)
    basemodel = BaseModelMock(random_state=42, cv_folds=3, n_jobs=None)

    # 3) Criar cromossoma (máscara + hiperparâmetros)
    rng = np.random.default_rng(42)
    M = X_train.shape[1]
    m_init = max(2, int(np.ceil(0.01 * M)))

    mask = init_feature_mask(M, m_init=m_init, rng=rng)
    mask = repair_if_zero(mask, rng=rng)

    # hiperparâmetros exemplo (mais tarde vais amostrar do param_grid do STREAMLINE)
    hyperparams = {"n_estimators": 200, "max_depth": 8, "min_samples_split": 2}

    chrom = Chromosome(feature_mask=mask, hyperparams=hyperparams)

    # 4) Fitness BA penalizada
    lam = 0.05
    fitness = evaluate_fitness_ba_penalized(basemodel, chrom, X_train, y_train, lam=lam)

    print("✅ Smoke test OK")
    print(f"Fitness = {fitness:.4f}")
    print(f"Selected features = {chrom.n_selected()} / {M}")
    print(f"Scoring metric = {basemodel.scoring_metric}")
    print(f"Lambda = {lam}")


if __name__ == "__main__":
    main()