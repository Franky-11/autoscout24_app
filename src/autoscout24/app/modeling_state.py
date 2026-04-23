import ast
import json

import pandas as pd
import streamlit as st

from autoscout24.modeling.config import FeatureSelectionConfig, TrainingConfig
from autoscout24.modeling.evaluation import EvaluationReport
from autoscout24.modeling.registry import LoadedPersistedRun, load_persisted_runs
from autoscout24.modeling.service import MODEL_LABELS

SCENARIO_LOG_COLUMNS = [
    "Pipe ID",
    "Run ID",
    "Features",
    "Anzahl Features",
    "Model",
    "Scaler",
    "Target",
    "PCA Active",
    "PCA Components",
    "CV Folds",
    "Test Size",
    "Model Parameters",
    "Train Time (s)",
    "R2 Score",
    "RMSE",
    "MAE",
    "CV RMSE",
    "CV MAE",
    "Best Baseline RMSE",
    "Artifact Path",
]

PREDICTION_LOG_COLUMNS = [
    "Pipe ID",
    "Run ID",
    "Marke",
    "Modell",
    "PS",
    "Erstzulassung",
    "Kilometerstand",
    "Kraftstoff",
    "Getriebe",
    "Angebotstyp",
    "Vorhergesagter Preis",
]


def initialize_modeling_state() -> None:
    defaults: dict[str, object] = {
        "model_training": False,
        "scenario_log": pd.DataFrame(columns=SCENARIO_LOG_COLUMNS),
        "car_price": pd.DataFrame(columns=PREDICTION_LOG_COLUMNS),
        "show_scenarios_popover": False,
        "features": None,
        "model": "",
        "scaler": "",
        "pca_check": False,
        "n_components": 0,
        "test_size": 0,
        "target_transform": "raw",
        "cv_folds": 3,
        "last_config_signature": None,
        "pipeline_store": {},
        "evaluation_report": None,
        "persisted_runs_loaded": False,
        "comparison_results": pd.DataFrame(),
        "comparison_context_signature": None,
        "setup_model_key": "lr",
        "setup_scaler_key": "Standard",
        "setup_target_transform": "raw",
        "setup_test_size_pct": 20,
        "setup_cv_folds": 3,
        "setup_pca_enabled": False,
        "setup_n_components": 10,
        "active_screening_candidate": None,
        "pending_screening_candidate": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.persisted_runs_loaded:
        (
            st.session_state.scenario_log,
            st.session_state.pipeline_store,
        ) = merge_persisted_runs(
            st.session_state.scenario_log,
            st.session_state.pipeline_store,
            load_persisted_runs(),
        )
        st.session_state.persisted_runs_loaded = True


def build_config_signature(
    feature_config: FeatureSelectionConfig,
    training_config: TrainingConfig,
    model_params: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "Features": sorted(feature_config.all_features),
        "Model": training_config.model_label,
        "Scaler": training_config.scaler_key,
        "Target": training_config.target_transform,
        "PCA Active": training_config.pca_enabled,
        "PCA Components": training_config.effective_pca_components,
        "Test Size": training_config.test_size,
        "CV Folds": training_config.cv_folds,
        "Model Parameters": _serialize_model_params(model_params),
    }


def config_exists(
    scenario_log: pd.DataFrame,
    feature_config: FeatureSelectionConfig,
    training_config: TrainingConfig,
    model_params: dict[str, object] | None = None,
) -> bool:
    current_config = build_config_signature(feature_config, training_config, model_params)

    for _, row in scenario_log.iterrows():
        row_config = {
            "Features": sorted(row["Features"]),
            "Model": row["Model"],
            "Scaler": row["Scaler"],
            "Target": row["Target"],
            "PCA Active": row["PCA Active"],
            "PCA Components": row["PCA Components"],
            "Test Size": row["Test Size"],
            "CV Folds": row["CV Folds"],
            "Model Parameters": _serialize_model_params(row["Model Parameters"]),
        }
        if current_config == row_config:
            return True

    return False


def merge_persisted_runs(
    scenario_log: pd.DataFrame,
    pipeline_store: dict[int, object],
    persisted_runs: list[LoadedPersistedRun],
) -> tuple[pd.DataFrame, dict[int, object]]:
    if not persisted_runs:
        return scenario_log, pipeline_store

    merged_log = scenario_log.copy()
    merged_store = dict(pipeline_store)
    existing_run_ids = set(merged_log["Run ID"].tolist())
    next_pipe_id = _next_pipe_id(merged_log)

    for persisted_run in persisted_runs:
        if persisted_run.run_id in existing_run_ids:
            continue

        merged_log = pd.concat(
            [
                merged_log,
                pd.DataFrame(
                    [
                        build_scenario_log_row(
                            persisted_run,
                            pipe_id=next_pipe_id,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
        merged_store[next_pipe_id] = persisted_run.pipeline
        existing_run_ids.add(persisted_run.run_id)
        next_pipe_id += 1

    return merged_log, merged_store


def build_scenario_log_row(
    persisted_run: LoadedPersistedRun,
    pipe_id: int,
) -> dict[str, object]:
    metadata = persisted_run.metadata
    feature_config = metadata.get("feature_config", {})
    training_config = metadata.get("training_config", {})
    features = [
        *feature_config.get("base_features", []),
        *feature_config.get("engineered_features", []),
    ]
    holdout_metrics = metadata.get("holdout_metrics", {})
    cross_validation = metadata.get("cross_validation", {})
    baselines = metadata.get("baselines", [])
    best_baseline_rmse = min((baseline["rmse"] for baseline in baselines), default=float("nan"))
    model_key = training_config.get("model_key", "")

    return {
        "Pipe ID": pipe_id,
        "Run ID": persisted_run.run_id,
        "Features": features,
        "Anzahl Features": len(metadata.get("feature_columns", [])),
        "Model": MODEL_LABELS.get(model_key, model_key),
        "Scaler": training_config.get("scaler_key", "None"),
        "Target": training_config.get("target_transform", "raw"),
        "PCA Active": training_config.get("pca_enabled", False),
        "PCA Components": (
            training_config.get("n_components", "N/A")
            if training_config.get("pca_enabled", False)
            else "N/A"
        ),
        "CV Folds": training_config.get("cv_folds", 3),
        "Test Size": training_config.get("test_size", float("nan")),
        "Model Parameters": str(metadata.get("model_parameters")),
        "Train Time (s)": metadata.get("train_time", float("nan")),
        "R2 Score": holdout_metrics.get("r2", float("nan")),
        "RMSE": holdout_metrics.get("rmse", float("nan")),
        "MAE": holdout_metrics.get("mae", float("nan")),
        "CV RMSE": cross_validation.get("mean_rmse", float("nan")),
        "CV MAE": cross_validation.get("mean_mae", float("nan")),
        "Best Baseline RMSE": best_baseline_rmse,
        "Artifact Path": str(persisted_run.run_dir),
    }


def restore_evaluation_report(metadata: dict[str, object]) -> EvaluationReport | None:
    evaluation_payload = metadata.get("evaluation_report")
    if not evaluation_payload:
        return None
    return EvaluationReport.from_dict(evaluation_payload)


def _next_pipe_id(scenario_log: pd.DataFrame) -> int:
    if scenario_log.empty:
        return 1
    return int(scenario_log["Pipe ID"].max()) + 1


def apply_screening_candidate(candidate: dict[str, object]) -> None:
    st.session_state.active_screening_candidate = {
        "candidate_label": candidate["Candidate Label"],
        "model_key": candidate["Model Key"],
        "model_label": candidate["Model"],
        "scaler_key": str(candidate["Scaler"]),
        "pca_enabled": bool(candidate["PCA"]),
        "n_components": (
            int(candidate["PCA Components"]) if candidate["PCA Components"] != "N/A" else 10
        ),
        "model_params": dict(candidate["Parameters Raw"]),
    }
    st.session_state.pending_screening_candidate = None


def queue_screening_candidate(candidate: dict[str, object]) -> None:
    st.session_state.pending_screening_candidate = candidate


def consume_pending_screening_candidate() -> bool:
    candidate = st.session_state.pending_screening_candidate
    if not candidate:
        return False

    apply_screening_candidate(candidate)
    return True


def clear_screening_candidate() -> None:
    st.session_state.active_screening_candidate = None
    st.session_state.pending_screening_candidate = None


def _serialize_model_params(model_params: object) -> str:
    parsed_params = model_params
    if isinstance(model_params, str):
        try:
            parsed_params = ast.literal_eval(model_params)
        except (SyntaxError, ValueError):
            parsed_params = model_params

    normalized_params = _normalize_model_params(parsed_params)
    return json.dumps(normalized_params, sort_keys=True, default=str)


def _normalize_model_params(model_params: object) -> object:
    if isinstance(model_params, dict):
        return {
            str(key): _normalize_model_params(value)
            for key, value in sorted(model_params.items(), key=lambda item: str(item[0]))
        }
    if isinstance(model_params, list | tuple):
        return [_normalize_model_params(value) for value in model_params]
    return model_params
