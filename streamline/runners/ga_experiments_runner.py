import copy
import json
import os
import shutil
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from streamline.runners.model_runner import ModelExperimentRunner


class GAExperimentsRunner:
    """
    Multi-phase experiment orchestrator for GA-based joint feature selection
    and hyperparameter optimization.

    Phase 1:
        Initial screening of operator combinations
        (selection + crossover + default mutation)
        using a reduced set of representative models and a small budget.

    Phase 2:
        Refinement of the best Phase 1 combinations by testing
        internal operator parameter values.

    Phase 3:
        Final run with the best refined configuration(s),
        more final models, a larger population, and optional early stopping.
    """

    def __init__(self, base_params: Dict[str, Any]):
        self.base_params = copy.deepcopy(base_params)

        self.output_path = self.base_params["output_path"]
        self.source_experiment_name = self.base_params["experiment_name"]
        self.source_experiment_path = os.path.join(
            self.output_path,
            self.source_experiment_name
        )

        if not os.path.exists(self.source_experiment_path):
            raise FileNotFoundError(
                f"Base experiment folder not found: {self.source_experiment_path}"
            )

        self.summary_root = os.path.join(
            self.output_path,
            f"{self.source_experiment_name}_ga_search"
        )
        os.makedirs(self.summary_root, exist_ok=True)

        # =========================================================
        # Models per phase
        # =========================================================
        self.phase1_models = self.base_params.get(
            "ga_phase1_models",
            ["LR", "RF", "SVM"]
        )
        self.final_models = self.base_params.get(
            "ga_final_models",
            ["LR", "RF", "SVM", "XGB", "ANN"]
        )

        self.phase1_top_k = int(self.base_params.get("ga_phase1_top_k", 3))
        self.phase2_top_k = int(self.base_params.get("ga_phase2_top_k", 2))

        # =========================================================
        # Infer number of features
        # =========================================================
        self.n_features = self._infer_n_features()
        print(f"[GAExperimentsRunner] Detected number of features: {self.n_features}")

        # =========================================================
        # Budget per phase
        # =========================================================
        self.phase1_n_generations = int(self.base_params.get("ga_phase1_n_generations", 6))
        self.phase2_n_generations = int(self.base_params.get("ga_phase2_n_generations", 12))
        self.phase3_n_generations = int(self.base_params.get("ga_phase3_n_generations", 50))

        self.phase1_population_size = int(
            self.base_params.get(
                "ga_phase1_population_size",
                self._default_phase1_population_size(self.n_features)
            )
        )
        self.phase2_population_size = int(
            self.base_params.get(
                "ga_phase2_population_size",
                self._default_phase2_population_size(self.n_features)
            )
        )
        self.phase3_population_size = int(
            self.base_params.get("ga_phase3_population_size", 100)
        )

        # =========================================================
        # Early stopping per phase
        # =========================================================
        # Phase 1: no early stopping
        self.phase1_early_stop = False

        # Phase 2: also disabled by default
        self.phase2_early_stop = bool(self.base_params.get("ga_phase2_early_stop", False))
        self.phase2_early_stop_patience = int(self.base_params.get("ga_phase2_early_stop_patience", 4))
        self.phase2_early_stop_min_delta = float(self.base_params.get("ga_phase2_early_stop_min_delta", 1e-3))
        self.phase2_early_stop_min_generations = int(self.base_params.get("ga_phase2_early_stop_min_generations", 6))

        # Phase 3: enabled by default
        self.phase3_early_stop = bool(self.base_params.get("ga_phase3_early_stop", True))
        self.phase3_early_stop_patience = int(self.base_params.get("ga_phase3_early_stop_patience", 7))
        self.phase3_early_stop_min_delta = float(self.base_params.get("ga_phase3_early_stop_min_delta", 5e-4))
        self.phase3_early_stop_min_generations = int(self.base_params.get("ga_phase3_early_stop_min_generations", 10))

        # =========================================================
        # Phase 1 search space
        # =========================================================
        self.phase1_selection_methods = self.base_params.get(
            "ga_phase1_selection_methods",
            ["tournament", "rank", "truncation"]
        )
        self.phase1_crossover_methods = self.base_params.get(
            "ga_phase1_crossover_methods",
            ["uniform", "one_point"]
        )

        self.phase1_mutation_profiles = [
            {
                "name": "default",
                "ga_p_flip": None,
                "ga_max_flips": None,
                "ga_p_hp_mut": 0.3,
                "ga_hp_mut_frac": 0.3,
                "ga_hp_mut_min": 1,
            }
        ]

        # =========================================================
        # Phase 2 refinement grids
        # =========================================================
        self.tournament_k_grid = self.base_params.get("ga_tournament_k_grid", [2, 3, 5])
        self.rank_eta_grid = self.base_params.get("ga_rank_eta_grid", [1.2, 1.7, 2.0])
        self.truncation_top_frac_grid = self.base_params.get("ga_truncation_top_frac_grid", [0.3, 0.5, 0.7])

        self.p_crossover_grid = self.base_params.get("ga_p_crossover_grid", [0.7, 0.9])
        self.p_swap_hp_grid = self.base_params.get("ga_p_swap_hp_grid", [0.3, 0.5, 0.7])

        self.phase2_mutation_profiles = [
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
            {
                "name": "aggressive",
                "ga_p_flip": None,
                "ga_max_flips": None,
                "ga_p_hp_mut": 0.5,
                "ga_hp_mut_frac": 0.5,
                "ga_hp_mut_min": 2,
            }
        ]

    # =========================================================
    # Public API
    # =========================================================
    def run(self):
        print("\n==============================")
        print("GA MULTI-PHASE SEARCH STARTED")
        print("==============================")

        print(f"[Phase 1] pop={self.phase1_population_size}, gens={self.phase1_n_generations}, early_stop={self.phase1_early_stop}")
        print(f"[Phase 2] pop={self.phase2_population_size}, gens={self.phase2_n_generations}, early_stop={self.phase2_early_stop}")
        print(f"[Phase 3] pop={self.phase3_population_size}, gens={self.phase3_n_generations}, early_stop={self.phase3_early_stop}")
        print()       
        
        print("Phase 1 search space:")
        phase1_ranked = self.run_phase1()
        top_phase1 = self.select_top_configs(phase1_ranked, top_k=self.phase1_top_k)

        print("\nPhase 2 search space:")
        phase2_ranked = self.run_phase2(top_phase1)
        top_phase2 = self.select_top_configs(phase2_ranked, top_k=self.phase2_top_k)

        print("\nPhase 3 search space:")
        phase3_ranked = self.run_phase3(top_phase2)

        self.save_global_summary(
            phase1_ranked=phase1_ranked,
            phase2_ranked=phase2_ranked,
            phase3_ranked=phase3_ranked
        )

        print("\n==============================")
        print("GA MULTI-PHASE SEARCH FINISHED")
        print("==============================")

        return phase3_ranked

    # =========================================================
    # Phase 1
    # =========================================================
    def run_phase1(self) -> List[Dict[str, Any]]:
        phase_name = "phase1"
        phase_dir = self._phase_dir(phase_name)
        results = []

        exp_id = 1
        for selection_method in self.phase1_selection_methods:
            for crossover_method in self.phase1_crossover_methods:
                for mutation_profile in self.phase1_mutation_profiles:
                    exp_cfg = {
                        "selection": {
                            "method": selection_method
                        },
                        "crossover": {
                            "mask_method": crossover_method,
                            "hp_method": "uniform",
                            "p_crossover": 0.9,
                            "p_swap_hp": 0.5,
                        },
                        "mutation": mutation_profile,
                        "elitism": int(self.base_params.get("ga_elitism", 2)),
                        "population_size": self.phase1_population_size,
                        "n_generations": self.phase1_n_generations,
                        "models": self.phase1_models,
                        "early_stop": False,
                    }

                    print(f"\n[Phase 1 - Experiment {exp_id}] Running with configuration:")
                    print(selection_method, crossover_method, mutation_profile["name"])
                    
                    params = self.build_params_for_experiment(
                        exp_cfg=exp_cfg,
                        phase_name=phase_name,
                        exp_id=exp_id
                    )

                    self.execute_experiment(params)

                    result = self.collect_experiment_results(
                        params=params,
                        exp_cfg=exp_cfg,
                        phase_name=phase_name,
                        exp_id=exp_id
                    )

                    results.append(result)
                    self._save_phase_result_row(phase_dir, result)
                    exp_id += 1

        ranked = self.rank_results(results)
        self._save_ranked_results(phase_dir, ranked, "phase1_ranked_results.csv")
        print("[Phase 1] Completed")
        return ranked

    # =========================================================
    # Phase 2
    # =========================================================
    def run_phase2(self, top_phase1_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        phase_name = "phase2"
        phase_dir = self._phase_dir(phase_name)
        results = []

        exp_id = 1
        for base_cfg in top_phase1_configs:
            refined_cfgs = self.expand_phase2_configs(base_cfg)

            for exp_cfg in refined_cfgs:
                exp_cfg["population_size"] = self.phase2_population_size
                exp_cfg["n_generations"] = self.phase2_n_generations
                exp_cfg["models"] = self.phase1_models
                exp_cfg["early_stop"] = self.phase2_early_stop

                params = self.build_params_for_experiment(
                    exp_cfg=exp_cfg,
                    phase_name=phase_name,
                    exp_id=exp_id
                )

                self.execute_experiment(params)

                result = self.collect_experiment_results(
                    params=params,
                    exp_cfg=exp_cfg,
                    phase_name=phase_name,
                    exp_id=exp_id
                )

                results.append(result)
                self._save_phase_result_row(phase_dir, result)
                exp_id += 1

        ranked = self.rank_results(results)
        self._save_ranked_results(phase_dir, ranked, "phase2_ranked_results.csv")
        print("[Phase 2] Completed")
        return ranked

    # =========================================================
    # Phase 3
    # =========================================================
    def run_phase3(self, top_phase2_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        phase_name = "phase3"
        phase_dir = self._phase_dir(phase_name)
        results = []

        exp_id = 1
        for base_cfg in top_phase2_configs:
            exp_cfg = copy.deepcopy(base_cfg)
            exp_cfg["models"] = self.final_models
            exp_cfg["population_size"] = self.phase3_population_size
            exp_cfg["n_generations"] = self.phase3_n_generations
            exp_cfg["early_stop"] = self.phase3_early_stop

            params = self.build_params_for_experiment(
                exp_cfg=exp_cfg,
                phase_name=phase_name,
                exp_id=exp_id
            )

            self.execute_experiment(params)

            result = self.collect_experiment_results(
                params=params,
                exp_cfg=exp_cfg,
                phase_name=phase_name,
                exp_id=exp_id
            )

            results.append(result)
            self._save_phase_result_row(phase_dir, result)
            exp_id += 1

        ranked = self.rank_results(results)
        self._save_ranked_results(phase_dir, ranked, "phase3_ranked_results.csv")
        
        print("[Phase 3] Completed")
        return ranked

    # =========================================================
    # Phase-specific config expansion
    # =========================================================
    def expand_phase2_configs(self, base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []

        selection_method = base_cfg["selection"]["method"]

        if selection_method == "tournament":
            selection_variants = [
                {"method": "tournament", "tournament_k": k}
                for k in self.tournament_k_grid
            ]
        elif selection_method == "rank":
            selection_variants = [
                {"method": "rank", "rank_eta": eta}
                for eta in self.rank_eta_grid
            ]
        elif selection_method == "truncation":
            selection_variants = [
                {"method": "truncation", "truncation_top_frac": frac}
                for frac in self.truncation_top_frac_grid
            ]
        else:
            selection_variants = [copy.deepcopy(base_cfg["selection"])]

        for selection_cfg in selection_variants:
            for p_crossover in self.p_crossover_grid:
                for p_swap_hp in self.p_swap_hp_grid:
                    for mutation_profile in self.phase2_mutation_profiles:
                        cfg = copy.deepcopy(base_cfg)

                        cfg["selection"] = selection_cfg
                        cfg["crossover"] = {
                            "mask_method": base_cfg["crossover"]["mask_method"],
                            "hp_method": "uniform",
                            "p_crossover": p_crossover,
                            "p_swap_hp": p_swap_hp,
                        }
                        cfg["mutation"] = mutation_profile
                        cfg["elitism"] = int(self.base_params.get("ga_elitism", 2))

                        configs.append(cfg)

        return configs

    # =========================================================
    # Build parameters for each experiment
    # =========================================================
    def build_params_for_experiment(
        self,
        exp_cfg: Dict[str, Any],
        phase_name: str,
        exp_id: int
    ) -> Dict[str, Any]:
        params = copy.deepcopy(self.base_params)

        params["experiment_name"] = self.make_experiment_name(
            phase_name=phase_name,
            exp_id=exp_id,
            exp_cfg=exp_cfg
        )

        params["do_model"] = True
        params["do_ga_opt"] = True

        # GA replaces classical feature selection
        params["do_feat_sel"] = False

        params["algorithms"] = exp_cfg["models"]
        params["exclude"] = []

        # Base GA budget
        params["ga_population_size"] = exp_cfg["population_size"]
        params["ga_n_generations"] = exp_cfg["n_generations"]
        params["ga_elitism"] = exp_cfg.get("elitism", 2)

        # Selection
        params["ga_selection_method"] = exp_cfg["selection"]["method"]
        params["ga_tournament_k"] = exp_cfg["selection"].get("tournament_k", 3)
        params["ga_rank_eta"] = exp_cfg["selection"].get("rank_eta", 1.7)
        params["ga_truncation_top_frac"] = exp_cfg["selection"].get("truncation_top_frac", 0.5)

        # Crossover
        params["ga_crossover_mask_method"] = exp_cfg["crossover"].get("mask_method", "uniform")
        params["ga_crossover_hp_method"] = exp_cfg["crossover"].get("hp_method", "uniform")
        params["ga_p_crossover"] = exp_cfg["crossover"].get("p_crossover", 0.9)
        params["ga_p_swap_hp"] = exp_cfg["crossover"].get("p_swap_hp", 0.5)

        # Mutation
        params["ga_p_flip"] = exp_cfg["mutation"].get("ga_p_flip", None)
        params["ga_max_flips"] = exp_cfg["mutation"].get("ga_max_flips", None)
        params["ga_p_hp_mut"] = exp_cfg["mutation"].get("ga_p_hp_mut", 0.3)
        params["ga_hp_mut_frac"] = exp_cfg["mutation"].get("ga_hp_mut_frac", 0.3)
        params["ga_hp_mut_min"] = exp_cfg["mutation"].get("ga_hp_mut_min", 1)

        # Early stopping
        if phase_name == "phase1":
            params["ga_early_stop"] = False
        elif phase_name == "phase2":
            params["ga_early_stop"] = self.phase2_early_stop
            params["ga_early_stop_patience"] = self.phase2_early_stop_patience
            params["ga_early_stop_min_delta"] = self.phase2_early_stop_min_delta
            params["ga_early_stop_min_generations"] = self.phase2_early_stop_min_generations
        elif phase_name == "phase3":
            params["ga_early_stop"] = self.phase3_early_stop
            params["ga_early_stop_patience"] = self.phase3_early_stop_patience
            params["ga_early_stop_min_delta"] = self.phase3_early_stop_min_delta
            params["ga_early_stop_min_generations"] = self.phase3_early_stop_min_generations

        return params

    # =========================================================
    # Experiment execution
    # =========================================================
    def execute_experiment(self, params: Dict[str, Any]) -> None:
        experiment_path = os.path.join(self.output_path, params["experiment_name"])

        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)

        self._clone_base_experiment_structure(
            source_path=self.source_experiment_path,
            target_path=experiment_path
        )

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
            ga_config=self._build_ga_config_from_params(params)
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

    # =========================================================
    # Folder structure / cloning
    # =========================================================
    def _clone_base_experiment_structure(self, source_path: str, target_path: str) -> None:
        shutil.copytree(source_path, target_path)

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
            os.makedirs(model_eval_dir, exist_ok=True)
            os.makedirs(os.path.join(model_eval_dir, "pickled_metrics"), exist_ok=True)

    def _list_dataset_dirs(self, experiment_path: str) -> List[str]:
        remove_list = {
            ".DS_Store",
            "metadata.pickle",
            "metadata.csv",
            "algInfo.pickle",
            "jobsCompleted",
            "logs",
            "jobs",
            "DatasetComparisons",
            "UsefulNotebooks",
            "dask_logs",
        }

        dataset_dirs = []
        for entry in os.listdir(experiment_path):
            if entry in remove_list:
                continue
            full = os.path.join(experiment_path, entry)
            if os.path.isdir(full):
                dataset_dirs.append(full)

        return dataset_dirs

    # =========================================================
    # Infer number of features
    # =========================================================
    def _infer_n_features(self) -> int:
        dataset_dirs = self._list_dataset_dirs(self.source_experiment_path)

        if not dataset_dirs:
            raise RuntimeError(
                f"Could not find any datasets in: {self.source_experiment_path}"
            )

        first_dataset_dir = dataset_dirs[0]

        candidate_files = [
            os.path.join(first_dataset_dir, "CVDatasets", "CV_0_Train.csv"),
            os.path.join(first_dataset_dir, "CVDatasets", "CV_1_Train.csv"),
            os.path.join(first_dataset_dir, "CVDatasets", "CV_0", "train.csv"),
            os.path.join(first_dataset_dir, "CVDatasets", "train.csv"),
        ]

        dataset_file = None
        for path in candidate_files:
            if os.path.exists(path):
                dataset_file = path
                break

        if dataset_file is None:
            cv_dir = os.path.join(first_dataset_dir, "CVDatasets")
            if os.path.exists(cv_dir):
                for file_name in os.listdir(cv_dir):
                    if file_name.lower().endswith(".csv") and "train" in file_name.lower():
                        dataset_file = os.path.join(cv_dir, file_name)
                        break

        if dataset_file is None:
            raise RuntimeError(
                f"Could not locate a training file to infer the number of features in {first_dataset_dir}"
            )

        df = pd.read_csv(dataset_file)

        drop_cols = []
        class_label = self.base_params.get("class_label", "Class")
        instance_label = self.base_params.get("instance_label", None)

        if class_label in df.columns:
            drop_cols.append(class_label)
        if instance_label and instance_label in df.columns:
            drop_cols.append(instance_label)

        df = df.drop(columns=drop_cols, errors="ignore")
        return int(df.shape[1])

    def _default_phase1_population_size(self, n_features: int) -> int:
        return int(min(40, max(20, int(0.25 * n_features))))

    def _default_phase2_population_size(self, n_features: int) -> int:
        return int(min(60, max(25, int(0.35 * n_features))))

    # =========================================================
    # Result collection
    # =========================================================
    def collect_experiment_results(
        self,
        params: Dict[str, Any],
        exp_cfg: Dict[str, Any],
        phase_name: str,
        exp_id: int
    ) -> Dict[str, Any]:
        experiment_path = os.path.join(self.output_path, params["experiment_name"])

        ga_rows = self._read_ga_metadata_rows(experiment_path)
        runtime_vals = self._read_runtime_values(experiment_path)

        result = {
            "phase": phase_name,
            "experiment_id": exp_id,
            "experiment_name": params["experiment_name"],
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
            fitness_vals = [
                row["final_fitness"]
                for row in ga_rows
                if row.get("final_fitness") is not None
            ]
            n_feat_vals = [
                row["n_selected_features"]
                for row in ga_rows
                if row.get("n_selected_features") is not None
            ]

            if fitness_vals:
                result["mean_final_fitness"] = float(np.mean(fitness_vals))

            if n_feat_vals:
                result["mean_n_selected_features"] = float(np.mean(n_feat_vals))

            alg_map = {}
            for row in ga_rows:
                alg = row.get("algorithm")
                fit = row.get("final_fitness")
                if alg is None or fit is None:
                    continue
                alg_map.setdefault(alg, []).append(float(fit))

            alg_mean_fitness = {
                alg: float(np.mean(vals))
                for alg, vals in alg_map.items()
                if len(vals) > 0
            }

            result["algorithm_mean_fitness"] = alg_mean_fitness

            if alg_mean_fitness:
                best_alg = max(alg_mean_fitness.items(), key=lambda x: x[1])[0]
                result["best_algorithm_by_fitness"] = best_alg

        details_path = os.path.join(
            self._phase_dir(phase_name),
            f"experiment_{exp_id:03d}_details.json"
        )
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
                if not filename.endswith("_ga_metadata.json"):
                    continue

                full_file = os.path.join(ga_meta_dir, filename)
                try:
                    with open(full_file, "r", encoding="utf-8") as f:
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
                if not filename.startswith("runtime_"):
                    continue

                full_file = os.path.join(runtime_dir, filename)
                try:
                    with open(full_file, "r", encoding="utf-8") as f:
                        runtimes.append(float(f.read().strip()))
                except Exception:
                    pass

        return runtimes

    # =========================================================
    # Ranking / summaries
    # =========================================================
    def rank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(r):
            fitness = -1e12 if r["mean_final_fitness"] is None else r["mean_final_fitness"]
            n_feat = 1e12 if r["mean_n_selected_features"] is None else r["mean_n_selected_features"]
            runtime = 1e12 if r["mean_runtime_sec"] is None else r["mean_runtime_sec"]

            return (fitness, -n_feat, -runtime)

        return sorted(results, key=sort_key, reverse=True)

    def select_top_configs(self, ranked_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return [copy.deepcopy(r["config"]) for r in ranked_results[:top_k]]

    def save_global_summary(
        self,
        phase1_ranked: List[Dict[str, Any]],
        phase2_ranked: List[Dict[str, Any]],
        phase3_ranked: List[Dict[str, Any]]
    ) -> None:
        summary = {
            "source_experiment_name": self.source_experiment_name,
            "n_features": self.n_features,
            "phase1_population_size": self.phase1_population_size,
            "phase2_population_size": self.phase2_population_size,
            "phase3_population_size": self.phase3_population_size,
            "phase1_best": phase1_ranked[0] if phase1_ranked else None,
            "phase2_best": phase2_ranked[0] if phase2_ranked else None,
            "phase3_best": phase3_ranked[0] if phase3_ranked else None,
        }

        summary_path = os.path.join(self.summary_root, "global_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(summary), f, indent=4, ensure_ascii=False)

    def make_experiment_name(self, phase_name: str, exp_id: int, exp_cfg: Dict[str, Any]) -> str:
        base_name = self.source_experiment_name

        sel = exp_cfg["selection"]["method"]
        cross = exp_cfg["crossover"]["mask_method"]
        mut = exp_cfg["mutation"].get("name", "mutation")

        name = f"{base_name}_{phase_name}_exp{exp_id:03d}_{sel}_{cross}_{mut}"

        if sel == "tournament":
            name += f"_k{exp_cfg['selection'].get('tournament_k', 3)}"
        elif sel == "rank":
            name += f"_eta{str(exp_cfg['selection'].get('rank_eta', 1.7)).replace('.', 'p')}"
        elif sel == "truncation":
            name += f"_top{str(exp_cfg['selection'].get('truncation_top_frac', 0.5)).replace('.', 'p')}"

        name += f"_pc{str(exp_cfg['crossover'].get('p_crossover', 0.9)).replace('.', 'p')}"
        name += f"_elite{exp_cfg.get('elitism', 2)}"

        return name

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