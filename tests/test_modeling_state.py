import pandas as pd

from autoscout24.app.modeling_state import build_scenario_log_row, merge_persisted_runs
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
