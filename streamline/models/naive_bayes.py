from abc import ABC
from streamline.modeling.basemodel import BaseModel
from streamline.modeling.parameters import get_parameters
from sklearn.naive_bayes import GaussianNB as NB


class NaiveBayesClassifier(BaseModel, ABC):
    model_name = "Naive Bayes"
    small_name = "NB"
    color = "silver"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None,
                 optimizer_backend='optuna', ga_config=None):
        super().__init__(NB, "Naive Bayes", cv_folds=cv_folds, scoring_metric=scoring_metric, metric_direction=metric_direction, random_state=random_state, cv=cv, n_jobs=n_jobs,
                         optimizer_backend=optimizer_backend, ga_config=ga_config)
        self.param_grid = get_parameters(self.model_name)
        self.small_name = "NB"
        self.color = "silver"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        self.params = {}
        mean_cv_score = self.hyper_eval()
        return mean_cv_score
