import os
import logging
import pickle
from pyexpat import model
import random
import time
import json
import numpy as np
import optuna
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit

from streamline.utils.job import Job


class ModelJob(Job):
    def __init__(self, full_path, output_path, experiment_name, cv_count, class_label="Class",
                 instance_label=None, scoring_metric='balanced_accuracy', metric_direction='maximize', n_trials=200,
                 timeout=900, training_subsample=0, uniform_fi=False, save_plot=False, random_state=None):
        """

        Args:
            full_path:
            output_path:
            experiment_name:
            cv_count:
            class_label:
            instance_label:
            scoring_metric:
            metric_direction:
            n_trials:
            timeout:
            uniform_fi:
            save_plot:
            random_state:
        """
        super().__init__()
        self.algorithm = ""
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        self.full_path = full_path
        self.cv_count = cv_count
        #self.data_name = self.full_path.split('/')[-1]
        self.data_name = os.path.basename(os.path.normpath(self.full_path))
        self.train_file_path = self.full_path + '/CVDatasets/' + self.data_name \
                               + '_CV_' + str(self.cv_count) + '_Train.csv'
        self.test_file_path = self.full_path + '/CVDatasets/' + self.data_name \
                              + '_CV_' + str(self.cv_count) + '_Test.csv'

        feature_names = pd.read_csv(self.train_file_path).columns.values.tolist()
        if self.instance_label is not None:
            feature_names.remove(self.instance_label)
        feature_names.remove(self.class_label)
        self.feature_names = feature_names

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 5 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 5 can begin")

        self.n_trials = n_trials
        self.timeout = timeout
        self.training_subsample = training_subsample
        self.random_state = random_state
        self.uniform_fi = uniform_fi
        self.feature_importance = None   
        self.pre_model_fi_scores= None   #FI to use to initialize the GA 
        self.save_plot = save_plot
        self.param_grid = None
    
    def load_pre_model_fi_scores(self):
        """
        Loads feature importance scores from phase 3 for the current fold.

        For now, Mutual Information is used as the FI source for GA initialization.
        The expected pickle structure follows phase 4:
            raw_data[0] -> array/vector of scores
            raw_data[1] -> dict mapping feature -> score
            raw_data[2] -> sorted ranking
        """
        fi_path = os.path.join(
            self.full_path,
            'feature_selection',
            'mutual_information',
            'pickledForPhase4',
            f'{self.cv_count}.pickle'
        )

        if not os.path.exists(fi_path):
            logging.warning(f'[GA] FI not found: {fi_path}')
            return None

        try:
            with open(fi_path, 'rb') as f:
                raw_data = pickle.load(f)

            scores = np.asarray(raw_data[0], dtype=float)

            if scores.shape[0] != len(self.feature_names):
                logging.warning(
                    f'[GA] Number of scores ({scores.shape[0]}) does not match number of model features '
                    f'({len(self.feature_names)}). GA will fall back to uniform initialization.'
                )
                return None

            return scores

        except Exception as e:
            logging.warning(f'[GA] Error loading FI from {fi_path}: {e}')
            return None
        
        
    def run(self, model):
        """

        Args:
            model: model object

        """
        self.job_start_time = time.time()  # for tracking phase runtime
        self.algorithm = model.small_name
        logging.info('Running ' + str(self.algorithm) + ' on ' + str(self.train_file_path))
        ret = self.run_model(model)

        # Pickle all evaluation metrics for ML model training and evaluation
        pickle.dump(ret, open(self.full_path
                              + '/model_evaluation/pickled_metrics/'
                              + self.algorithm + '_CV_' + str(self.cv_count) + "_metrics.pickle", 'wb'))

        # Save runtime of ml algorithm training and evaluation
        self.save_runtime()

        # Print phase completion
        #logging.info(self.full_path.split('/')[-1] + " [CV_" + str(self.cv_count) + "] (" + self.algorithm
        #            + ") training complete. ------------------------------------")
        #experiment_path = '/'.join(self.full_path.split('/')[:-1])
        dataset_name = os.path.basename(os.path.normpath(self.full_path))
        logging.info(dataset_name + " [CV_" + str(self.cv_count) + "] (" + self.algorithm
             + ") training complete. ------------------------------------")
        
        dataset_name = os.path.basename(os.path.normpath(self.full_path))
        experiment_path = os.path.dirname(self.full_path)
        job_file = open(os.path.join(
            experiment_path,
            'jobsCompleted',
            'job_model_' + dataset_name + '_' + str(self.cv_count) + '_' + self.algorithm + '.txt'
        ), 'w')
        job_file.write('complete')
        job_file.close()
        # experiment_path = os.path.dirname(self.full_path)
        # job_file = open(experiment_path + '/jobsCompleted/job_model_' + self.full_path.split('/')[-1]
        #                 + '_' + str(self.cv_count) + '_' + self.algorithm + '.txt', 'w')
        # job_file.write('complete')
        # job_file.close()

    def run_model(self, model):
        """

        Args:
            model: model object

        Returns: list of metrics [metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas]

        """
        # Set random seeds for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        # Load training and testing datasets separating features from outcome for scikit-learn-based modeling
        x_train, y_train, x_test, y_test = self.data_prep()

        self.pre_model_fi_scores = None
        if getattr(model, 'optimizer_backend', None) == 'ga':
            self.pre_model_fi_scores = self.load_pre_model_fi_scores()

        model.fit(x_train, y_train, self.n_trials, self.timeout, self.feature_names, fi_scores=self.pre_model_fi_scores)
       
        print("\n[DEBUG] Depois de model.fit()")
        # print(f"[DEBUG] optimizer_backend = {getattr(model, 'optimizer_backend', None)}")
        # print(f"[DEBUG] has ga_best_individual = {hasattr(model, 'ga_best_individual')}")
        # print(f"[DEBUG] ga_best_individual is None = {getattr(model, 'ga_best_individual', None) is None}")
        # print(f"[DEBUG] has ga_history = {hasattr(model, 'ga_history')}")
        # print(f"[DEBUG] selected_feature_indices = {getattr(model, 'selected_feature_indices', None)}")
        # print(f"[DEBUG] params = {getattr(model, 'params', None)}")
            
        if 0 < self.training_subsample < x_train.shape[0] and model.small_name in ['XGB', 'SVM', 'ANN', 'KNN']:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.training_subsample, random_state=self.random_state)
            for train_index, _ in sss.split(x_train, y_train):
                x_train = x_train[train_index]
                y_train = y_train[train_index]
            logging.warning('For ' + model.small_name
                            + ', training sample reduced to ' + str(x_train.shape[0]) + ' instances')

        if not os.path.exists(self.full_path + '/models/'):
            os.makedirs(self.full_path + '/models/')

        if not model.is_single:
            if self.save_plot and model.optimizer_backend=='optuna' and model.study is not None:
                try:
                    fig = optuna.visualization.plot_parallel_coordinate(model.study)
                    fig.write_image(self.full_path + '/models/' + self.algorithm +
                                    '_ParamOptimization_' + str(self.cv_count) + '.png')
                except Exception as e:
                    logging.warning(str(e))
                    logging.warning('Warning: Optuna Optimization Visualization Generation Failed for '
                                    'Due to Known Release Issue.  '
                                    'Please install Optuna 2.0.0 to avoid this issue.')
            # Print results and hyperparamter values for best hyperparameter sweep trial
            self.export_best_params(self.full_path + '/models/' + self.algorithm +
                                    '_bestparams' + str(self.cv_count) + '.csv',
                                    model.params)
        else:  # Specify hyperparameter values (no sweep)
            self.export_best_params(self.full_path + '/models/' + self.algorithm +
                                    '_usedparams' + str(self.cv_count) + '.csv',
                                    model.params)
        self.export_ga_metadata(model)

        if self.uniform_fi:
            results = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=self.random_state,
                                             scoring=self.scoring_metric)
            self.feature_importance = results.importances_mean
        else:
            try:
                self.feature_importance = model.model.feature_importances_
            except AttributeError:
                results = permutation_importance(model, x_train, y_train, n_repeats=10,
                                                 random_state=self.random_state,
                                                 scoring=self.scoring_metric)
                self.feature_importance = results.importances_mean

        if not os.path.exists(self.full_path + '/models/pickledModels/'):
            os.makedirs(self.full_path + '/models/pickledModels/')

        with open(self.full_path + '/models/pickledModels/' + self.algorithm +
                  '_' + str(self.cv_count) + '.pickle', 'wb') as file:
            pickle.dump(model, file)

        metric_list, fpr, tpr, roc_auc, prec, recall, \
            prec_rec_auc, ave_prec, probas_ = model.model_evaluation(x_test, y_test)
        fi = self.feature_importance

        return [metric_list, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

    def data_prep(self):
        """
        Loads target cv training dataset, separates class from features and removes instance labels.
        """
        train = pd.read_csv(self.train_file_path)
        test = pd.read_csv(self.test_file_path)
        if self.instance_label is not None:
            train = train.drop(self.instance_label, axis=1)
            test = test.drop(self.instance_label, axis=1)
        x_train = train.drop(self.class_label, axis=1).values
        y_train = train[self.class_label].values
        x_test = test.drop(self.class_label, axis=1).values
        y_test = test[self.class_label].values
        del train  # memory cleanup
        del test  # memory cleanup
        return x_train, y_train, x_test, y_test

    def save_runtime(self):
        """
        Save ML algorithm training and evaluation runtime for this phase.
        """
        runtime_file = open(self.full_path + '/runtime/runtime_' + self.algorithm + '_CV' + str(self.cv_count) + '.txt',
                            'w')
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()

    @staticmethod
    def _json_safe(value):
        """
        Converte tipos numpy/pandas para tipos serializáveis em JSON.
        """
        if isinstance(value, dict):
            return {k: ModelJob._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ModelJob._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [ModelJob._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    # ALTERAÇÃO GA: exportação explícita da metadata do melhor indivíduo e histórico do GA
    def export_ga_metadata(self, model):
        """
        Guarda metadata explícita do GA, caso o modelo tenha sido otimizado com backend GA.
        """
        # print("\n[DEBUG] Entrou em export_ga_metadata()")
        # print(f"[DEBUG] optimizer_backend = {getattr(model, 'optimizer_backend', None)}")
        # print(f"[DEBUG] has ga_best_individual = {hasattr(model, 'ga_best_individual')}")
        # print(f"[DEBUG] ga_best_individual is None = {getattr(model, 'ga_best_individual', None) is None}")

        if not hasattr(model, 'optimizer_backend') or model.optimizer_backend != 'ga':
            return

        if not hasattr(model, 'ga_best_individual') or model.ga_best_individual is None:
            return

        ga_metadata_dir = self.full_path + '/models/ga_metadata'
        ga_history_dir = self.full_path + '/models/ga_history'

        if not os.path.exists(ga_metadata_dir):
            os.makedirs(ga_metadata_dir)

        if not os.path.exists(ga_history_dir):
            os.makedirs(ga_history_dir)

        best = model.ga_best_individual

        selected_mask = None
        if hasattr(model, 'selected_features_mask') and model.selected_features_mask is not None:
            selected_mask = np.asarray(model.selected_features_mask).astype(int).tolist()

        selected_indices = None
        if hasattr(model, 'selected_feature_indices') and model.selected_feature_indices is not None:
            selected_indices = np.asarray(model.selected_feature_indices).astype(int).tolist()

        if selected_indices is None and selected_mask is not None:
            selected_indices = [i for i, v in enumerate(selected_mask) if v == 1]

        selected_feature_names = None
        if selected_indices is not None:
            selected_feature_names = [self.feature_names[i] for i in selected_indices]

        n_selected_features = len(selected_indices) if selected_indices is not None else None

        final_fitness = getattr(best, 'fitness', None)
        fitness_meta = getattr(best, 'meta', None)
        fail_reason = getattr(best, 'fail_reason', None)

        metadata = {
            'algorithm': self.algorithm,
            'cv_fold': self.cv_count,
            'dataset': self.data_name,
            'selected_features_mask': selected_mask,
            'selected_feature_indices': selected_indices,
            'selected_feature_names': selected_feature_names,
            'n_selected_features': n_selected_features,
            'best_hyperparameters': getattr(model, 'params', None),
            'final_fitness': final_fitness,
            'fitness_meta': fitness_meta,
            'fail_reason': fail_reason
        }

        metadata = self._json_safe(metadata)
        metadata_path = ga_metadata_dir + '/' + self.algorithm + '_CV_' + str(self.cv_count) + '_ga_metadata.json'
        
        # print(f"[DEBUG] Vai guardar metadata em: {metadata_path}")

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        if hasattr(model, 'ga_history') and model.ga_history is not None:
            try:
                history_df = pd.DataFrame(model.ga_history)
                history_df = history_df.applymap(self._json_safe)
                # print(f"[DEBUG] Vai guardar history em: {ga_history_dir}")
                history_df.to_csv(
                    ga_history_dir + '/' + self.algorithm + '_CV_' + str(self.cv_count) + '_ga_history.csv',
                    index=False
                )
            except Exception as e:
                logging.warning('GA history export failed: ' + str(e))



    @staticmethod
    def export_best_params(file_name, param_grid):
        """
        Exports the best hyperparameter scores to output file.
        """
        best_params_copy = dict(param_grid)
        for best in best_params_copy:
            best_params_copy[best] = [best_params_copy[best]]
        df = pd.DataFrame.from_dict(best_params_copy)
        df.to_csv(file_name, index=False)
