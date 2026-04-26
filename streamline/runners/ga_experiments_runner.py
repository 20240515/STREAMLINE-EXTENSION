import copy
import json
import os
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from streamline.runners.model_runner import ModelExperimentRunner


class GAExperimentsRunner:
    """
    Dataset-aware GA runner for STREAMLINE.

    Instead of doing a large search over all possible GA operator families and their parameters,
    this runner first classifies the dataset by dimensionality and fixes the GA
    operator family:

        p << n  -> Tournament + One-point(mask)/ Uniform(hyperparameters) + Bit-flip/Hyperparameter resampling
        p < n   -> Tournament + Uniform(mask)/ Uniform(hyperparameters) + Bit-flip/Hyperparameter resampling
        p >= n  -> Rank-based + Uniform(mask)/ Uniform(hyperparameters) + Bit-flip/Hyperparameter resampling
        p >> n  -> Rank-based + Uniform(mask)/ Uniform(hyperparameters) + Bit-flip/Hyperparameter resampling

    Then it only calibrates internal parameters of that chosen operator family
    and performs a final run over the final model list to select the best model.
    """

    def __init__(self, base_params: Dict[str, Any]):
        """Initialize the GAExperimentsRunner with base parameters."""
        
        self.base_params = copy.deepcopy(base_params)
        self.output_path = self.base_params["output_path"]
        self.source_experiment_name = self.base_params["experiment_name"]
        self.source_experiment_path = os.path.join(self.output_path, self.source_experiment_name)

        if not os.path.exists(self.source_experiment_path):
            raise FileNotFoundError(f"Base experiment folder not found: {self.source_experiment_path}")

        dataset_dirs = self._list_dataset_dirs(self.source_experiment_path)
        if not dataset_dirs:
            raise RuntimeError(
                f"Could not find any datasets in: {self.source_experiment_path}"
            )

        self.dataset_root = dataset_dirs[0]

        self.summary_root = os.path.join(
            self.dataset_root,
            f"ga_dataset_aware_search",
        )
        os.makedirs(self.summary_root, exist_ok=True)

        self.temp_runs_root = os.path.join(
            self.source_experiment_path,
            "ga_temp_runs",
        )
        os.makedirs(self.temp_runs_root, exist_ok=True)

        # self.calibration_models = self.base_params.get(
        #     "ga_calibration_models",
        #     self.base_params.get("ga_phase1_models", ["LR", "RF", "XGB"]),
        # )

        self.calibration_models = self.base_params.get(
            "ga_calibration_models",
            ["LR", "RF", "NB"],
        )
        
        self.final_models = self.base_params.get(
            "ga_final_models",
            ["LR", "RF", "SVM", "XGB", "ANN"],
        )

        self.n_samples, self.n_features = self._infer_dataset_shape()
        self.p_to_n_ratio = self.n_features / max(1, self.n_samples)
        self.dataset_regime = self._classify_dataset_regime(self.n_samples, self.n_features)
        self.base_operator_config = self._operator_config_for_regime(self.dataset_regime)

        print("[GAExperimentsRunner] Dataset-aware GA search")
        print(f"[GAExperimentsRunner] n_samples={self.n_samples}")
        print(f"[GAExperimentsRunner] n_features={self.n_features}")
        print(f"[GAExperimentsRunner] p/n={self.p_to_n_ratio:.4f}")
        print(f"[GAExperimentsRunner] regime={self.dataset_regime}")
        print(f"[GAExperimentsRunner] chosen operators={self._short_config_label(self.base_operator_config)}")

        self.calibration_n_generations = int(
            self.base_params.get("ga_calibration_n_generations", 12)
        )

        self.final_n_generations = int(
            self.base_params.get("ga_final_n_generations", 50)
        )

        self.calibration_population_size = int(
            self.base_params.get(
                "ga_calibration_population_size",
                self._default_calibration_population_size(self.n_features),
            )
        )

        self.final_population_size = int(
            self.base_params.get(
                "ga_final_population_size",
                self._default_final_population_size(self.n_features),
            )
        )

        self.calibration_top_k = int(
            self.base_params.get("ga_calibration_top_k", 1)
        )

        self.calibration_early_stop = bool(
            self.base_params.get("ga_calibration_early_stop", False)
        )
        self.calibration_early_stop_patience = int(
            self.base_params.get("ga_calibration_early_stop_patience", 4)
        )
        self.calibration_early_stop_min_delta = float(
            self.base_params.get("ga_calibration_early_stop_min_delta", 1e-3)
        )
        self.calibration_early_stop_min_generations = int(
            self.base_params.get("ga_calibration_early_stop_min_generations", 6)
        )

        self.final_early_stop = bool(
            self.base_params.get("ga_final_early_stop", True)
        )
        self.final_early_stop_patience = int(
            self.base_params.get("ga_final_early_stop_patience", 7)
        )
        self.final_early_stop_min_delta = float(
            self.base_params.get("ga_final_early_stop_min_delta", 5e-4)
        )
        self.final_early_stop_min_generations = int(
            self.base_params.get("ga_final_early_stop_min_generations", 10)
        )

        # Compact calibration grids: by default this gives few experiments, not dozens/hundreds.
        self.tournament_k_grid = self.base_params.get("ga_tournament_k_grid", [2, 3])
        self.rank_eta_grid = self.base_params.get("ga_rank_eta_grid", [1.3, 1.7])
        self.truncation_top_frac_grid = self.base_params.get("ga_truncation_top_frac_grid", [0.5])
        self.p_crossover_grid = self.base_params.get("ga_p_crossover_grid", [0.8, 0.9])
        self.p_swap_hp_grid = self.base_params.get("ga_p_swap_hp_grid", [0.5])

        self.calibration_mutation_profiles = self.base_params.get(
            "ga_calibration_mutation_profiles",
            [
                {
                    "name": "mild",
                    "ga_p_flip": None,
                    "ga_max_flips": None,
                    "ga_p_hp_mut": 0.2,
                    "ga_hp_mut_frac": 0.2,
                    "ga_hp_mut_min": 1,
                },
                {
                    "name": "default",
                    "ga_p_flip": None,
                    "ga_max_flips": None,
                    "ga_p_hp_mut": 0.3,
                    "ga_hp_mut_frac": 0.3,
                    "ga_hp_mut_min": 1,
                },
            ],
        )

        # self.phase2_n_generations = int(
        #     self.base_params.get("ga_phase2_n_generations", self.base_params.get("ga_calibration_n_generations", 12))
        # )
        # self.final_n_generations = int(
        #     self.base_params.get("ga_final_n_generations", self.base_params.get("ga_phase3_n_generations", 50))
        # )

        # self.phase2_population_size = int(
        #     self.base_params.get(
        #         "ga_phase2_population_size",
        #         self.base_params.get("ga_calibration_population_size", self._default_calibration_population_size(self.n_features)),
        #     )
        # )
        # self.final_population_size = int(
        #     self.base_params.get(
        #         "ga_final_population_size",
        #         self.base_params.get("ga_phase3_population_size", self._default_final_population_size(self.n_features)),
        #     )
        # )

        # self.phase2_top_k = int(self.base_params.get("ga_phase2_top_k", 1))

        # self.phase2_early_stop = bool(self.base_params.get("ga_phase2_early_stop", False))
        # self.phase2_early_stop_patience = int(self.base_params.get("ga_phase2_early_stop_patience", 4))
        # self.phase2_early_stop_min_delta = float(self.base_params.get("ga_phase2_early_stop_min_delta", 1e-3))
        # self.phase2_early_stop_min_generations = int(self.base_params.get("ga_phase2_early_stop_min_generations", 6))

        # self.final_early_stop = bool(self.base_params.get("ga_final_early_stop", self.base_params.get("ga_phase3_early_stop", True)))
        # self.final_early_stop_patience = int(self.base_params.get("ga_final_early_stop_patience", self.base_params.get("ga_phase3_early_stop_patience", 7)))
        # self.final_early_stop_min_delta = float(self.base_params.get("ga_final_early_stop_min_delta", self.base_params.get("ga_phase3_early_stop_min_delta", 5e-4)))
        # self.final_early_stop_min_generations = int(self.base_params.get("ga_final_early_stop_min_generations", self.base_params.get("ga_phase3_early_stop_min_generations", 10)))

        # # Compact grids: by default this gives 8 experiments, not dozens/hundreds.
        # self.tournament_k_grid = self.base_params.get("ga_tournament_k_grid", [2, 3])
        # self.rank_eta_grid = self.base_params.get("ga_rank_eta_grid", [1.3, 1.7])
        # self.truncation_top_frac_grid = self.base_params.get("ga_truncation_top_frac_grid", [0.5])
        # self.p_crossover_grid = self.base_params.get("ga_p_crossover_grid", [0.8, 0.9])
        # self.p_swap_hp_grid = self.base_params.get("ga_p_swap_hp_grid", [0.5])
        # self.phase2_mutation_profiles = self.base_params.get(
        #     "ga_phase2_mutation_profiles",
        #     [
        #         {"name": "mild", "ga_p_flip": None, "ga_max_flips": None, "ga_p_hp_mut": 0.2, "ga_hp_mut_frac": 0.2, "ga_hp_mut_min": 1},
        #         {"name": "default", "ga_p_flip": None, "ga_max_flips": None, "ga_p_hp_mut": 0.3, "ga_hp_mut_frac": 0.3, "ga_hp_mut_min": 1},
        #     ],
        # )

    def run(self):
        print("\n========================================")
        print("GA DATASET-AWARE SEARCH STARTED")
        print("========================================")
        print(f"[Dataset] regime={self.dataset_regime}, p/n={self.p_to_n_ratio:.4f}")
        print(
            f"[Calibration] pop={self.calibration_population_size}, "
            f"gens={self.calibration_n_generations}, "
            f"models={self.calibration_models}"
        )
        print(
            f"[Final modelling] pop={self.final_population_size}, "
            f"gens={self.final_n_generations}, "
            f"models={self.final_models}"
        )

        calibration_ranked = self.run_calibration([self.base_operator_config])
        top_calibration = self.select_top_configs(
            calibration_ranked,
            top_k=self.calibration_top_k
        )
        final_ranked = self.run_final(top_calibration)

        self.copy_best_final_outputs(final_ranked)

        self.save_global_summary(
            calibration_ranked=calibration_ranked,
            final_ranked=final_ranked
        )

        print("\n========================================")
        print("GA DATASET-AWARE SEARCH FINISHED")
        print("========================================")
        return final_ranked

    def run_calibration(self, base_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        phase_name = "calibration"
        phase_dir = self._phase_dir(phase_name)
        results = []
        exp_id = 1

        for base_cfg in base_configs:
            for exp_cfg in self.expand_calibration_configs(base_cfg):
                exp_cfg["population_size"] = self.calibration_population_size
                exp_cfg["n_generations"] = self.calibration_n_generations
                exp_cfg["models"] = self.calibration_models
                exp_cfg["early_stop"] = self.calibration_early_stop

                print(f"\n[Calibration - Experiment {exp_id}] {self._short_config_label(exp_cfg)}")
                params = self.build_params_for_experiment(exp_cfg, phase_name, exp_id)
                self.execute_experiment(params)
                result = self.collect_experiment_results(params, exp_cfg, phase_name, exp_id)
                results.append(result)
                self._save_phase_result_row(phase_dir, result)
                exp_id += 1

        ranked = self.rank_results(results)
        self._save_ranked_results(
            phase_dir,
            ranked,
            "calibration_ranked_results.csv"
        )
        print("[Calibration] Completed")
        return ranked


    def run_final(self, top_calibration_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        phase_name = "final"
        phase_dir = self._phase_dir(phase_name)
        results = []
        exp_id = 1

        for base_cfg in top_calibration_configs:
            exp_cfg = copy.deepcopy(base_cfg)
            exp_cfg["models"] = self.final_models
            exp_cfg["population_size"] = self.final_population_size
            exp_cfg["n_generations"] = self.final_n_generations
            exp_cfg["early_stop"] = self.final_early_stop

            print(f"\n[Final - Experiment {exp_id}] {self._short_config_label(exp_cfg)}")
            params = self.build_params_for_experiment(exp_cfg, phase_name, exp_id)
            self.execute_experiment(params)

            result = self.collect_experiment_results(params, exp_cfg, phase_name, exp_id)
            results.append(result)
            self._save_phase_result_row(phase_dir, result)
            exp_id += 1

        ranked = self.rank_results(results)
        self._save_ranked_results(phase_dir, ranked, "final_ranked_results.csv")
        print("[Final] Completed")
        return ranked

    def _classify_dataset_regime(self, n_samples: int, n_features: int) -> str:
        ratio = n_features / max(1, n_samples)
        much_less_ratio = float(self.base_params.get("ga_p_much_less_than_n_ratio", 0.10))
        much_greater_ratio = float(self.base_params.get("ga_p_much_greater_than_n_ratio", 10.0))

        if ratio <= much_less_ratio:
            return "p_much_less_than_n"
        if n_features < n_samples:
            return "p_less_than_n"
        if ratio >= much_greater_ratio:
            return "p_much_greater_than_n"
        return "p_greater_equal_n"

    def _operator_config_for_regime(self, regime: str) -> Dict[str, Any]:
        if regime == "p_much_less_than_n":
            selection = {"method": "tournament"}
            crossover = {"mask_method": "one_point", "hp_method": "uniform", "p_crossover": 0.9, "p_swap_hp": 0.5}
        elif regime == "p_less_than_n":
            selection = {"method": "tournament"}
            crossover = {"mask_method": "uniform", "hp_method": "uniform", "p_crossover": 0.9, "p_swap_hp": 0.5}
        elif regime in {"p_greater_equal_n", "p_much_greater_than_n"}:
            selection = {"method": "rank"}
            crossover = {"mask_method": "uniform", "hp_method": "uniform", "p_crossover": 0.9, "p_swap_hp": 0.5}
        else:
            raise ValueError(f"Unknown dataset regime: {regime}")

        return {
            "dataset_regime": regime,
            "selection": selection,
            "crossover": crossover,
            "mutation": {"name": "default", "ga_p_flip": None, "ga_max_flips": None, "ga_p_hp_mut": 0.3, "ga_hp_mut_frac": 0.3, "ga_hp_mut_min": 1},
            "elitism": int(self.base_params.get("ga_elitism", 2)),
        }

    def expand_calibration_configs(self, base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []
        method = base_cfg["selection"]["method"]

        if method == "tournament":
            selection_variants = [
                {"method": "tournament", "tournament_k": int(k)}
                for k in self.tournament_k_grid
            ]
        elif method == "rank":
            selection_variants = [
                {"method": "rank", "rank_eta": float(eta)}
                for eta in self.rank_eta_grid
            ]
        elif method == "truncation":
            selection_variants = [
                {"method": "truncation", "truncation_top_frac": float(frac)}
                for frac in self.truncation_top_frac_grid
            ]
        else:
            selection_variants = [copy.deepcopy(base_cfg["selection"])]

        for selection_cfg in selection_variants:
            for p_crossover in self.p_crossover_grid:
                for p_swap_hp in self.p_swap_hp_grid:
                    for mutation_profile in self.calibration_mutation_profiles:
                        cfg = copy.deepcopy(base_cfg)
                        cfg["selection"] = copy.deepcopy(selection_cfg)
                        cfg["crossover"] = {
                            "mask_method": base_cfg["crossover"]["mask_method"],
                            "hp_method": "uniform",
                            "p_crossover": float(p_crossover),
                            "p_swap_hp": float(p_swap_hp),
                        }
                        cfg["mutation"] = copy.deepcopy(mutation_profile)
                        cfg["elitism"] = int(self.base_params.get("ga_elitism", 2))
                        configs.append(cfg)

        return configs

    def build_params_for_experiment(self, exp_cfg: Dict[str, Any], phase_name: str, exp_id: int) -> Dict[str, Any]:
        params = copy.deepcopy(self.base_params)
        params["output_path"] = self.temp_runs_root
        params["experiment_name"] = self.make_experiment_name(phase_name, exp_id, exp_cfg)
        params["do_model"] = True
        params["do_ga_opt"] = True
        params["do_feat_sel"] = False
        params["algorithms"] = exp_cfg["models"]
        params["exclude"] = []

        params["ga_population_size"] = exp_cfg["population_size"]
        params["ga_n_generations"] = exp_cfg["n_generations"]
        params["ga_elitism"] = exp_cfg.get("elitism", 2)

        params["ga_selection_method"] = exp_cfg["selection"]["method"]
        params["ga_tournament_k"] = exp_cfg["selection"].get("tournament_k", 3)
        params["ga_rank_eta"] = exp_cfg["selection"].get("rank_eta", 1.7)
        params["ga_truncation_top_frac"] = exp_cfg["selection"].get("truncation_top_frac", 0.5)

        params["ga_crossover_mask_method"] = exp_cfg["crossover"].get("mask_method", "uniform")
        params["ga_crossover_hp_method"] = exp_cfg["crossover"].get("hp_method", "uniform")
        params["ga_p_crossover"] = exp_cfg["crossover"].get("p_crossover", 0.9)
        params["ga_p_swap_hp"] = exp_cfg["crossover"].get("p_swap_hp", 0.5)

        params["ga_p_flip"] = exp_cfg["mutation"].get("ga_p_flip", None)
        params["ga_max_flips"] = exp_cfg["mutation"].get("ga_max_flips", None)
        params["ga_p_hp_mut"] = exp_cfg["mutation"].get("ga_p_hp_mut", 0.3)
        params["ga_hp_mut_frac"] = exp_cfg["mutation"].get("ga_hp_mut_frac", 0.3)
        params["ga_hp_mut_min"] = exp_cfg["mutation"].get("ga_hp_mut_min", 1)

        if phase_name == "calibration":
            params["ga_early_stop"] = self.calibration_early_stop
            params["ga_early_stop_patience"] = self.calibration_early_stop_patience
            params["ga_early_stop_min_delta"] = self.calibration_early_stop_min_delta
            params["ga_early_stop_min_generations"] = self.calibration_early_stop_min_generations
        elif phase_name == "final":
            params["ga_early_stop"] = self.final_early_stop
            params["ga_early_stop_patience"] = self.final_early_stop_patience
            params["ga_early_stop_min_delta"] = self.final_early_stop_min_delta
            params["ga_early_stop_min_generations"] = self.final_early_stop_min_generations
        else:
            params["ga_early_stop"] = False
        return params

    def execute_experiment(self, params: Dict[str, Any]) -> None:
        experiment_path = os.path.join(
            params["output_path"],
            params["experiment_name"],
        )

        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)

        self._clone_base_experiment_structure(self.source_experiment_path, experiment_path)
        self._clear_previous_model_outputs(experiment_path)

        model_runner = ModelExperimentRunner(
            params["output_path"],
            params["experiment_name"],
            algorithms=params["algorithms"],
            exclude=params["exclude"],
            class_label=params["class_label"],
            instance_label=params["instance_label"],
            scoring_metric=params["primary_metric"],
            metric_direction=params["metric_direction"],
            training_subsample=params["training_subsample"],
            use_uniform_fi=params["use_uniform_fi"],
            n_trials=params["n_trials"],
            timeout=params["timeout"],
            save_plots=False,
            do_lcs_sweep=params["do_lcs_sweep"],
            lcs_nu=params["lcs_nu"],
            lcs_n=params["lcs_n"],
            lcs_iterations=params["lcs_iterations"],
            lcs_timeout=params["lcs_timeout"],
            resubmit=params["model_resubmit"],
            random_state=params["random_state"],
            n_jobs=params["n_jobs"],
            run_cluster=params["run_cluster"],
            queue=params["queue"],
            reserved_memory=params["reserved_memory"],
            do_ga_opt=True,
            ga_config=self._build_ga_config_from_params(params),
        )
        model_runner.run(run_parallel=params["run_parallel"])

    def _build_ga_config_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "population_size": params["ga_population_size"],
            "n_generations": params["ga_n_generations"],
            "elitism": params["ga_elitism"],
            "selection": {
                "method": params["ga_selection_method"],
                "tournament_k": params["ga_tournament_k"],
                "rank_eta": params["ga_rank_eta"],
                "truncation_top_frac": params["ga_truncation_top_frac"],
            },
            "crossover": {
                "mask_method": params["ga_crossover_mask_method"],
                "hp_method": params["ga_crossover_hp_method"],
                "p_crossover": params["ga_p_crossover"],
                "p_swap_hp": params["ga_p_swap_hp"],
            },
            "mutation": {
                "p_flip": params["ga_p_flip"],
                "max_flips": params["ga_max_flips"],
                "p_hp_mut": params["ga_p_hp_mut"],
                "hp_mut_frac": params["ga_hp_mut_frac"],
                "hp_mut_min": params["ga_hp_mut_min"],
            },
            "early_stop": params.get("ga_early_stop", False),
            "early_stop_patience": params.get("ga_early_stop_patience", 5),
            "early_stop_min_delta": params.get("ga_early_stop_min_delta", 1e-3),
            "early_stop_min_generations": params.get("ga_early_stop_min_generations", 5),
        }

    def _clone_base_experiment_structure(self, source_path: str, target_path: str) -> None:
        shutil.copytree(
            source_path,
            target_path,
            ignore=shutil.ignore_patterns(
                "ga_temp_runs",
                "ga_dataset_aware_search",
            ),
        )
    def _clear_previous_model_outputs(self, experiment_path: str) -> None:
        for dataset_dir in self._list_dataset_dirs(experiment_path):
            models_dir = os.path.join(dataset_dir, "models")
            model_eval_dir = os.path.join(dataset_dir, "model_evaluation")
            if os.path.exists(models_dir):
                shutil.rmtree(models_dir)
            if os.path.exists(model_eval_dir):
                shutil.rmtree(model_eval_dir)
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(os.path.join(models_dir, "pickledModels"), exist_ok=True)
            os.makedirs(os.path.join(models_dir, "ga_metadata"), exist_ok=True)
            os.makedirs(os.path.join(models_dir, "ga_history"), exist_ok=True)
            os.makedirs(model_eval_dir, exist_ok=True)
            os.makedirs(os.path.join(model_eval_dir, "pickled_metrics"), exist_ok=True)

    def _list_dataset_dirs(self, experiment_path: str) -> List[str]:
        remove_list = {
            ".DS_Store", "metadata.pickle", "metadata.csv", "algInfo.pickle",
            "jobsCompleted", "logs", "jobs", "DatasetComparisons",
            "UsefulNotebooks", "dask_logs",
        }
        return [
            os.path.join(experiment_path, entry)
            for entry in os.listdir(experiment_path)
            if entry not in remove_list and os.path.isdir(os.path.join(experiment_path, entry))
        ]

    def _infer_dataset_shape(self) -> Tuple[int, int]:
        dataset_dirs = self._list_dataset_dirs(self.source_experiment_path)
        if not dataset_dirs:
            raise RuntimeError(f"Could not find any datasets in: {self.source_experiment_path}")
        first_dataset_dir = dataset_dirs[0]
        dataset_file = self._find_training_file(first_dataset_dir)
        if dataset_file is None:
            raise RuntimeError(f"Could not locate a training file to infer dataset shape in {first_dataset_dir}")

        df = pd.read_csv(dataset_file)
        drop_cols = []
        class_label = self.base_params.get("class_label", "Class")
        instance_label = self.base_params.get("instance_label", None)
        if class_label in df.columns:
            drop_cols.append(class_label)
        if instance_label and instance_label in df.columns:
            drop_cols.append(instance_label)
        return int(df.shape[0]), int(df.drop(columns=drop_cols, errors="ignore").shape[1])

    def _find_training_file(self, dataset_dir: str) -> str:
        dataset_name = os.path.basename(dataset_dir)
        candidate_files = [
            os.path.join(dataset_dir, "CVDatasets", f"{dataset_name}_CV_0_Train.csv"),
            os.path.join(dataset_dir, "CVDatasets", f"{dataset_name}_CVOnly_0_Train.csv"),
            os.path.join(dataset_dir, "CVDatasets", "CV_0_Train.csv"),
            os.path.join(dataset_dir, "CVDatasets", "CV_1_Train.csv"),
            os.path.join(dataset_dir, "CVDatasets", "CV_0", "train.csv"),
            os.path.join(dataset_dir, "CVDatasets", "train.csv"),
        ]
        for path in candidate_files:
            if os.path.exists(path):
                return path
        cv_dir = os.path.join(dataset_dir, "CVDatasets")
        if os.path.exists(cv_dir):
            for file_name in sorted(os.listdir(cv_dir)):
                lower = file_name.lower()
                if lower.endswith(".csv") and "train" in lower:
                    return os.path.join(cv_dir, file_name)
        return None

    def _infer_n_features(self) -> int:
        return self._infer_dataset_shape()[1]

    def _default_calibration_population_size(self, n_features: int) -> int:
        return int(min(60, max(25, int(0.35 * n_features))))

    def _default_final_population_size(self, n_features: int) -> int:
        return int(min(100, max(40, int(0.50 * n_features))))

    def collect_experiment_results(self, params: Dict[str, Any], exp_cfg: Dict[str, Any], phase_name: str, exp_id: int) -> Dict[str, Any]:
        experiment_path = os.path.join(params["output_path"], params["experiment_name"])
        ga_rows = self._read_ga_metadata_rows(experiment_path)
        runtime_vals = self._read_runtime_values(experiment_path)

        result = {
            "phase": phase_name,
            "experiment_id": exp_id,
            "experiment_name": params["experiment_name"],
            "dataset_regime": self.dataset_regime,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "p_to_n_ratio": self.p_to_n_ratio,
            "config": exp_cfg,
            "n_ga_records": len(ga_rows),
            "mean_final_fitness": None,
            "mean_n_selected_features": None,
            "mean_runtime_sec": None,
            "best_algorithm_by_fitness": None,
            "algorithm_mean_fitness": {},
        }

        if runtime_vals:
            result["mean_runtime_sec"] = float(np.mean(runtime_vals))
        if ga_rows:
            fitness_vals = [row["final_fitness"] for row in ga_rows if row.get("final_fitness") is not None]
            n_feat_vals = [row["n_selected_features"] for row in ga_rows if row.get("n_selected_features") is not None]
            if fitness_vals:
                result["mean_final_fitness"] = float(np.mean(fitness_vals))
            if n_feat_vals:
                result["mean_n_selected_features"] = float(np.mean(n_feat_vals))

            alg_map = {}
            for row in ga_rows:
                alg = row.get("algorithm")
                fit = row.get("final_fitness")
                if alg is not None and fit is not None:
                    alg_map.setdefault(alg, []).append(float(fit))
            result["algorithm_mean_fitness"] = {alg: float(np.mean(vals)) for alg, vals in alg_map.items() if vals}
            if result["algorithm_mean_fitness"]:
                result["best_algorithm_by_fitness"] = max(result["algorithm_mean_fitness"].items(), key=lambda x: x[1])[0]

        details_path = os.path.join(self._phase_dir(phase_name), f"experiment_{exp_id:03d}_details.json")
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(result), f, indent=4, ensure_ascii=False)
        return self._json_safe(result)

    def _read_ga_metadata_rows(self, experiment_path: str) -> List[Dict[str, Any]]:
        rows = []
        for dataset_dir in self._list_dataset_dirs(experiment_path):
            ga_meta_dir = os.path.join(dataset_dir, "models", "ga_metadata")
            if not os.path.exists(ga_meta_dir):
                continue
            for filename in os.listdir(ga_meta_dir):
                if filename.endswith("_ga_metadata.json"):
                    try:
                        with open(os.path.join(ga_meta_dir, filename), "r", encoding="utf-8") as f:
                            rows.append(json.load(f))
                    except Exception:
                        pass
        return rows

    def _read_runtime_values(self, experiment_path: str) -> List[float]:
        runtimes = []
        for dataset_dir in self._list_dataset_dirs(experiment_path):
            runtime_dir = os.path.join(dataset_dir, "runtime")
            if not os.path.exists(runtime_dir):
                continue
            for filename in os.listdir(runtime_dir):
                if filename.startswith("runtime_"):
                    try:
                        with open(os.path.join(runtime_dir, filename), "r", encoding="utf-8") as f:
                            runtimes.append(float(f.read().strip()))
                    except Exception:
                        pass
        return runtimes

    def rank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(r):
            fitness = -1e12 if r["mean_final_fitness"] is None else r["mean_final_fitness"]
            n_feat = 1e12 if r["mean_n_selected_features"] is None else r["mean_n_selected_features"]
            runtime = 1e12 if r["mean_runtime_sec"] is None else r["mean_runtime_sec"]
            return fitness, -n_feat, -runtime
        return sorted(results, key=sort_key, reverse=True)

    def select_top_configs(self, ranked_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return [copy.deepcopy(r["config"]) for r in ranked_results[:top_k]]
    
    def copy_best_final_outputs(self, final_ranked: List[Dict[str, Any]]) -> None:
        if not final_ranked:
            return

        best_experiment_name = final_ranked[0]["experiment_name"]
        dataset_name = os.path.basename(self.dataset_root)

        source_dataset_dir = os.path.join(
            self.temp_runs_root,
            best_experiment_name,
            dataset_name,
        )

        target_dataset_dir = self.dataset_root

        source_models = os.path.join(source_dataset_dir, "models")
        target_models = os.path.join(target_dataset_dir, "models")

        source_model_eval = os.path.join(source_dataset_dir, "model_evaluation")
        target_model_eval = os.path.join(target_dataset_dir, "model_evaluation")

        source_runtime = os.path.join(source_dataset_dir, "runtime")
        target_runtime = os.path.join(target_dataset_dir, "runtime")

        source_alginfo = os.path.join(self.temp_runs_root, best_experiment_name, "algInfo.pickle")
        target_alginfo = os.path.join(self.source_experiment_path, "algInfo.pickle")

        if not os.path.exists(source_models):
            raise FileNotFoundError(f"Source models folder not found: {source_models}")

        if not os.path.exists(source_model_eval):
            raise FileNotFoundError(f"Source model_evaluation folder not found: {source_model_eval}")

        if not os.path.exists(source_runtime):
            raise FileNotFoundError(f"Source runtime folder not found: {source_runtime}")
    
        if os.path.exists(target_models):
            shutil.rmtree(target_models)
        shutil.copytree(source_models, target_models)

        if os.path.exists(target_model_eval):
            shutil.rmtree(target_model_eval)
        shutil.copytree(source_model_eval, target_model_eval)

        if os.path.exists(target_runtime):
            shutil.rmtree(target_runtime)
        shutil.copytree(source_runtime, target_runtime)

        if os.path.exists(source_alginfo):
            shutil.copy2(source_alginfo, target_alginfo)

    def save_global_summary(
        self,
        calibration_ranked: List[Dict[str, Any]],
        final_ranked: List[Dict[str, Any]]
    ) -> None:
        summary = {
            "source_experiment_name": self.source_experiment_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "p_to_n_ratio": self.p_to_n_ratio,
            "dataset_regime": self.dataset_regime,
            "chosen_base_operator_config": self.base_operator_config,
            "calibration_population_size": self.calibration_population_size,
            "final_population_size": self.final_population_size,
            "calibration_best": calibration_ranked[0] if calibration_ranked else None,
            "final_best": final_ranked[0] if final_ranked else None,
        }

        with open(os.path.join(self.summary_root, "global_summary.json"), "w", encoding="utf-8") as f:
            json.dump(self._json_safe(summary), f, indent=4, ensure_ascii=False)

    def make_experiment_name(self, phase_name: str, exp_id: int, exp_cfg: Dict[str, Any]) -> str:
        
        sel = exp_cfg["selection"]["method"]
        cross = exp_cfg["crossover"]["mask_method"]
        mut = exp_cfg["mutation"].get("name", "mutation")
        regime = exp_cfg.get("dataset_regime", self.dataset_regime)
        name = f"{phase_name}_exp{exp_id:03d}_{regime}_{sel}_{cross}_{mut}"
        if sel == "tournament":
            name += f"_k{exp_cfg['selection'].get('tournament_k', 3)}"
        elif sel == "rank":
            name += f"_eta{str(exp_cfg['selection'].get('rank_eta', 1.7)).replace('.', 'p')}"
        elif sel == "truncation":
            name += f"_top{str(exp_cfg['selection'].get('truncation_top_frac', 0.5)).replace('.', 'p')}"
        name += f"_pc{str(exp_cfg['crossover'].get('p_crossover', 0.9)).replace('.', 'p')}"
        name += f"_hp{str(exp_cfg['crossover'].get('p_swap_hp', 0.5)).replace('.', 'p')}"
        name += f"_elite{exp_cfg.get('elitism', 2)}"
        return name

    def _short_config_label(self, exp_cfg: Dict[str, Any]) -> str:
        sel = exp_cfg["selection"]
        cross = exp_cfg["crossover"]
        mut = exp_cfg["mutation"]
        if sel["method"] == "tournament":
            sel_label = f"tournament(k={sel.get('tournament_k', 3)})"
        elif sel["method"] == "rank":
            sel_label = f"rank(eta={sel.get('rank_eta', 1.7)})"
        elif sel["method"] == "truncation":
            sel_label = f"truncation(top={sel.get('truncation_top_frac', 0.5)})"
        else:
            sel_label = sel["method"]
        return (
            f"regime={exp_cfg.get('dataset_regime', self.dataset_regime)}, "
            f"selection={sel_label}, crossover={cross.get('mask_method', 'uniform')}, "
            f"p_crossover={cross.get('p_crossover', 0.9)}, "
            f"p_swap_hp={cross.get('p_swap_hp', 0.5)}, mutation={mut.get('name', 'mutation')}"
        )

    def _phase_dir(self, phase_name: str) -> str:
        phase_dir = os.path.join(self.summary_root, phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        return phase_dir

    def _save_phase_result_row(self, phase_dir: str, result: Dict[str, Any]) -> None:
        csv_path = os.path.join(phase_dir, "results_incremental.csv")
        row = copy.deepcopy(result)
        row["config"] = json.dumps(row["config"], ensure_ascii=False)
        row["algorithm_mean_fitness"] = json.dumps(row["algorithm_mean_fitness"], ensure_ascii=False)
        df_row = pd.DataFrame([row])
        if os.path.exists(csv_path):
            df_row.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df_row.to_csv(csv_path, index=False)

    def _save_ranked_results(self, phase_dir: str, ranked_results: List[Dict[str, Any]], filename: str) -> None:
        rows = []
        for r in ranked_results:
            row = copy.deepcopy(r)
            row["config"] = json.dumps(row["config"], ensure_ascii=False)
            row["algorithm_mean_fitness"] = json.dumps(row["algorithm_mean_fitness"], ensure_ascii=False)
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(phase_dir, filename), index=False)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {k: GAExperimentsRunner._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [GAExperimentsRunner._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [GAExperimentsRunner._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value
