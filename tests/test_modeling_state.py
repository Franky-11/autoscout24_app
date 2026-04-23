import pandas as pd

from autoscout24.app.modeling_state import (
    SCENARIO_LOG_COLUMNS,
    build_config_signature,
    build_scenario_log_row,
    config_exists,
    merge_persisted_runs,
)
from autoscout24.modeling.config import FeatureSelectionConfig, TrainingConfig
from autoscout24.modeling.registry import LoadedPersistedRun


def test_merge_persisted_runs_appends_new_runs_without_duplicates(tmp_path):
    persisted_run = LoadedPersistedRun(
        run_id="run-001",
        run_dir=tmp_path / "run-001",
        metadata_path=tmp_path / "run-001" / "metadata.json",
        pipeline_path=tmp_path / "run-001" / "pipeline.joblib",
        metadata={
            "feature_config": {"base_features": ["hp", "make"], "engineered_features": ["car_age"]},
            "training_config": {
                "model_key": "lr",
                "scaler_key": "Standard",
                "pca_enabled": False,
                "n_components": 10,
                "test_size": 0.2,
                "target_transform": "raw",
                "cv_folds": 3,
            },
            "feature_columns": ["hp", "car_age", "make_BMW"],
            "holdout_metrics": {"rmse": 1000.0, "r2": 0.8, "mae": 700.0},
            "cross_validation": {"mean_rmse": 1100.0, "mean_mae": 750.0},
            "baselines": [{"rmse": 1500.0}],
            "model_parameters": {"fit_intercept": True},
            "train_time": 0.4,
        },
        pipeline={"model": "dummy"},
    )

    scenario_log, pipeline_store = merge_persisted_runs(
        scenario_log=pd.DataFrame(columns=["Run ID", "Pipe ID"]),
        pipeline_store={},
        persisted_runs=[persisted_run, persisted_run],
    )

    assert len(scenario_log) == 1
    assert scenario_log.iloc[0]["Run ID"] == "run-001"
    assert pipeline_store[1] == {"model": "dummy"}


def test_build_scenario_log_row_maps_metadata_to_ui_columns(tmp_path):
    persisted_run = LoadedPersistedRun(
        run_id="run-002",
        run_dir=tmp_path / "run-002",
        metadata_path=tmp_path / "run-002" / "metadata.json",
        pipeline_path=tmp_path / "run-002" / "pipeline.joblib",
        metadata={
            "feature_config": {"base_features": ["hp"], "engineered_features": ["car_age"]},
            "training_config": {
                "model_key": "rf",
                "scaler_key": "None",
                "pca_enabled": False,
                "test_size": 0.2,
                "target_transform": "log1p",
                "cv_folds": 4,
            },
            "feature_columns": ["hp", "car_age"],
            "holdout_metrics": {"rmse": 900.0, "r2": 0.82, "mae": 600.0},
            "cross_validation": {"mean_rmse": 950.0, "mean_mae": 620.0},
            "baselines": [{"rmse": 1500.0}, {"rmse": 1200.0}],
            "model_parameters": {"n_estimators": 250},
        },
        pipeline={"model": "rf"},
    )

    row = build_scenario_log_row(persisted_run, pipe_id=3)

    assert row["Pipe ID"] == 3
    assert row["Model"] == "Random Forest"
    assert row["Best Baseline RMSE"] == 1200.0
    assert row["Features"] == ["hp", "car_age"]


def test_build_config_signature_includes_model_params():
    feature_config = FeatureSelectionConfig(
        base_features=["hp", "mileage"],
        engineered_features=["car_age"],
    )
    training_config = TrainingConfig(
        model_key="cat",
        scaler_key="None",
        pca_enabled=False,
        n_components=10,
        test_size=0.2,
        target_transform="raw",
        cv_folds=3,
    )

    baseline_signature = build_config_signature(feature_config, training_config, None)
    candidate_signature = build_config_signature(
        feature_config,
        training_config,
        {"depth": 6, "iterations": 300},
    )

    assert baseline_signature != candidate_signature


def test_config_exists_matches_saved_candidate_params():
    feature_config = FeatureSelectionConfig(
        base_features=["hp", "mileage"],
        engineered_features=["car_age"],
    )
    training_config = TrainingConfig(
        model_key="cat",
        scaler_key="None",
        pca_enabled=False,
        n_components=10,
        test_size=0.2,
        target_transform="raw",
        cv_folds=3,
    )
    scenario_log = pd.DataFrame(
        [
            {
                "Pipe ID": 1,
                "Run ID": "run-001",
                "Features": feature_config.all_features,
                "Anzahl Features": 3,
                "Model": training_config.model_label,
                "Scaler": training_config.scaler_key,
                "Target": training_config.target_transform,
                "PCA Active": training_config.pca_enabled,
                "PCA Components": training_config.effective_pca_components,
                "CV Folds": training_config.cv_folds,
                "Test Size": training_config.test_size,
                "Model Parameters": "{'iterations': 300, 'depth': 6}",
                "Train Time (s)": 1.0,
                "R2 Score": 0.8,
                "RMSE": 1000.0,
                "MAE": 800.0,
                "CV RMSE": 1100.0,
                "CV MAE": 850.0,
                "Best Baseline RMSE": 1400.0,
                "Artifact Path": "/tmp/run-001",
            }
        ],
        columns=SCENARIO_LOG_COLUMNS,
    )

    assert config_exists(
        scenario_log,
        feature_config,
        training_config,
        {"depth": 6, "iterations": 300},
    )
    assert not config_exists(
        scenario_log,
        feature_config,
        training_config,
        {"depth": 8, "iterations": 300},
    )
