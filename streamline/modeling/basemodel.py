import copy
import logging
import optuna
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc
from streamline.utils.evaluation import class_eval
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score

#new imports for GA
from streamline.optimization import GeneticOptimizer, GeneticOptimizerConfig

import warnings
warnings.filterwarnings(action='ignore', module='sklearn')
warnings.filterwarnings(action='ignore', module='scipy')
warnings.filterwarnings(action='ignore', module='optuna')
warnings.filterwarnings(action="ignore", category=ConvergenceWarning, module="sklearn")


class BaseModel:
    def __init__(self, model, model_name,
                 cv_folds=3, scoring_metric='balanced_accuracy', metric_direction='maximize',
                 random_state=None, cv=None, sampler=None, n_jobs=None,
                 optimizer_backend='optuna', ga_config=None):
        """
        Base Model Class for all ML Models

        Args:
            model:
            model_name:
            cv_folds:
            scoring_metric:
            metric_direction:
            random_state:
            cv:
            sampler:
            n_jobs:
            optimizer_backend: 'optuna' or 'ga'
            ga_config:
        """
        self.is_single = True
        if model is not None:
            self.model = model()

        self.small_name = model_name.replace(" ", "_")
        self.model_name = model_name
        self.y_train = None
        self.x_train = None
        self.param_grid = None
        self.params = None
        self.random_state = random_state
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction

        if cv is None:
            self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            self.cv = cv

        self.n_jobs = n_jobs
        

        self.optimizer_backend = optimizer_backend.lower().strip()
        if self.optimizer_backend not in ['optuna', 'ga']:
            raise ValueError(f"Unsupported optimizer_backend: {self.optimizer_backend}")


        self.sampler = sampler
        if self.optimizer_backend == 'optuna':
            if sampler is None:
                self.sampler = optuna.samplers.TPESampler(seed=self.random_state)
            else:
                self.sampler = sampler

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.study = None
            
        self.ga_config = ga_config

        # if sampler is None:
        #     self.sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # else:
        #     self.sampler = sampler
        # self.study = None
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        # self.n_jobs = n_jobs

        #metadata do GA
        self.selected_features_mask = None
        self.selected_feature_indices = None
        self.ga_best_individual = None
        self.ga_history = None
        self.fi_scores = None

        print("DEBUG BaseModel:")
        print("  model_name =", self.model_name)
        print("  random_state =", self.random_state, type(self.random_state))
        print("  sampler =", self.sampler, type(self.sampler))
        print("  n_jobs =", self.n_jobs, type(self.n_jobs))
        print("  optimizer_backend =", self.optimizer_backend)



    def objective(self, trial, params=None):
        """
        Unimplemented objective function stub, needs to be overridden
        Args:
            trial: optuna trial object
            params: dict of optional params or None
        """
        raise NotImplementedError
    
    #new
    def _optimize_ga(self, x_train, y_train, fi_scores=None):

        rng = np.random.default_rng(self.random_state)

        ga = GeneticOptimizer(cfg=self.ga_config or GeneticOptimizerConfig())
        
        best_individual, history = ga.optimize(
            basemodel=self,
            X_train=x_train,
            y_train=y_train,
            rng=rng,
            fi_scores=fi_scores,
            cache_seed=self.random_state
        )

        self.ga_best_individual = best_individual
        self.ga_history = history
        self.is_single = False

        self.selected_features_mask = np.asarray(best_individual.feature_mask, dtype=bool)
        self.selected_feature_indices = np.where(self.selected_features_mask)[0]
        self.params = copy.deepcopy(best_individual.hyperparams)

        self.model = copy.deepcopy(self.model).set_params(**self.params) 

    # NEW
    def _select_features_if_needed(self, x):
        """
        Apply GA-selected feature subset when available.
        """
        if self.optimizer_backend == 'ga' and self.selected_features_mask is not None:
            return x[:, self.selected_features_mask]
        return x
    
    # NEW
    def run_optimization(self, x_train, y_train, n_trails, timeout, feature_names=None, fi_scores=None):
        """
        Dispatcher: chooses optuna or GA without changing optimize().
        """
        self.fi_scores = fi_scores

        if self.optimizer_backend == 'ga':
            self._optimize_ga(x_train, y_train, fi_scores=fi_scores)
        else:
            self.optimize(x_train, y_train, n_trails, timeout, feature_names)


    @ignore_warnings(category=ConvergenceWarning)
    def optimize(self, x_train, y_train, n_trails, timeout, feature_names=None):
        """
        Common model optimization function

        Args:
            x_train: train data
            y_train: label data
            n_trails: number of optuna trials
            timeout: maximum time for optuna trial timeout
            feature_names: header/name of features

        """
        self.x_train = x_train
        self.y_train = y_train
        for key, value in self.param_grid.items():
            if len(value) > 1 and key != 'expert_knowledge':
                self.is_single = False
                break

        if not self.is_single:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.study = optuna.create_study(direction=self.metric_direction, sampler=self.sampler)
            if self.model_name in ["Extreme Gradient Boosting", "Light Gradient Boosting"]:
                pos_inst = sum(y_train)
                neg_inst = len(y_train) - pos_inst
                class_weight = neg_inst / float(pos_inst)
                self.study.optimize(lambda trial: self.objective(trial, params={'class_weight': class_weight}),
                                    n_trials=n_trails, timeout=timeout,
                                    catch=(ValueError,))
            elif self.model_name == "Genetic Programming":
                self.study.optimize(lambda trial: self.objective(trial, params={'feature_names': feature_names}),
                                    n_trials=n_trails, timeout=timeout,
                                    catch=(ValueError,))
            else:
                self.study.optimize(lambda trial: self.objective(trial), n_trials=n_trails, timeout=timeout,
                                    catch=(ValueError,))

            logging.info('Best trial:')
            best_trial = self.study.best_trial
            logging.info('  Value: ' + str(best_trial.value))
            logging.info('  Params: ')
            for key, value in best_trial.params.items():
                logging.info('    {}: {}'.format(key, value))
            # Specify model with optimized hyperparameters
            # Export final model hyperparamters to csv file
            self.params = best_trial.params
            self.model = copy.deepcopy(self.model).set_params(**best_trial.params)
        else:
            self.params = copy.deepcopy(self.param_grid)
            for key, value in self.param_grid.items():
                self.params[key] = value[0]
            self.model = copy.deepcopy(self.model).set_params(**self.params)

    def feature_importance(self):
        """
        Unimplemented feature importance function stub
        """
        raise NotImplementedError

    def hyper_eval(self):
        """
        Hyper eval for objective function
        Returns: Returns hyper eval for objective function
        """
        logging.debug("Trial Parameters" + str(self.params))
        try:
            model = copy.deepcopy(self.model).set_params(**self.params)
            mean_cv_score = cross_val_score(model, self.x_train, self.y_train,
                                            scoring=self.scoring_metric,
                                            cv=self.cv, n_jobs=self.n_jobs).mean()
        except Exception as e:
            logging.error("KeyError while copying model " + self.model_name)
            logging.error(str(e))
            model_class = self.model.__class__
            model = model_class(**self.params)
            mean_cv_score = cross_val_score(model, self.x_train, self.y_train,
                                            scoring=self.scoring_metric,
                                            cv=self.cv, n_jobs=self.n_jobs).mean()
        logging.debug("Trail Completed")
        return mean_cv_score
    
    
    def model_evaluation(self, x_test, y_test):
        """
        Runs commands to gather all evaluations for later summaries and plots.
        """
        # Apply GA feature subset if needed
        x_test = self._select_features_if_needed(x_test)

        # Prediction evaluation
        y_pred = self.model.predict(x_test)
        metric_list = class_eval(y_test, y_pred)
        # Determine probabilities of class predictions for each test instance
        # (this will be used much later in calculating an ROC curve)
        probas_ = self.model.predict_proba(x_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        # Compute Precision/Recall curve and AUC
        prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
        prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
        prec_rec_auc = auc(recall, prec)
        ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])
        return metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_


    # def model_evaluation(self, x_test, y_test):
    #     """
    #     Runs commands to gather all evaluations for later summaries and plots.
    #     """
    #     # Prediction evaluation
    #     y_pred = self.model.predict(x_test)
    #     metric_list = class_eval(y_test, y_pred)
    #     # Determine probabilities of class predictions for each test instance
    #     # (this will be used much later in calculating an ROC curve)
    #     probas_ = self.model.predict_proba(x_test)
    #     # Compute ROC curve and area the curve
    #     fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    #     roc_auc = auc(fpr, tpr)
    #     # Compute Precision/Recall curve and AUC
    #     prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    #     prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    #     prec_rec_auc = auc(recall, prec)
    #     ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])
    #     return metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_

    # def fit(self, x_train, y_train, n_trails, timeout, feature_names=None):
    #     """
    #     Caller function to optimize
    #     """
    #     self.optimize(x_train, y_train, n_trails, timeout, feature_names)
    #     self.model.fit(x_train, y_train)

    def fit(self, x_train, y_train, n_trails, timeout, feature_names=None, fi_scores=None):
        """
        Caller function to optimize
        """
        self.run_optimization(
            x_train, y_train, n_trails, timeout,
            feature_names=feature_names,
            fi_scores=fi_scores
        )

        x_train_used = self._select_features_if_needed(x_train)
        self.model.fit(x_train_used, y_train)    

    def predict(self, x_in):
        """
        Function to predict with trained model
        Args:
            x_in: input data

        Returns: predictions y_pred
        """
        x_used = self._select_features_if_needed(x_in)
        return self.model.predict(x_used)        
        

    # def predict(self, x_in):
    #     """
    #     Function to predict with trained model
    #     Args:
    #         x_in: input data

    #     Returns: predictions y_pred

    #     """
    #     return self.model.predict(x_in)
