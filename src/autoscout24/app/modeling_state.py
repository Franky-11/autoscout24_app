import pandas as pd
import streamlit as st

from autoscout24.modeling.config import FeatureSelectionConfig, TrainingConfig

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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_config_signature(
    feature_config: FeatureSelectionConfig,
    training_config: TrainingConfig,
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
    }


def config_exists(
    scenario_log: pd.DataFrame,
    feature_config: FeatureSelectionConfig,
    training_config: TrainingConfig,
) -> bool:
    current_config = build_config_signature(feature_config, training_config)

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
        }
        if current_config == row_config:
            return True

    return False
