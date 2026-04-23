from dataclasses import dataclass
from itertools import islice

import pandas as pd
from sklearn.model_selection import ParameterGrid

from autoscout24.modeling.service import (
    MODEL_LABELS,
    rebuild_prepared_data_for_model,
    run_cross_validation,
)

MODEL_SEARCH_SPACE = {
    "lr": [
        {
            "scaler_selection": ["Standard", "MinMax"],
            "pca_check": [False],
            "model_params": [{"fit_intercept": True}],
        },
        {
            "scaler_selection": ["Standard"],
            "pca_check": [True],
            "n_components": [5, 10],
            "model_params": [{"fit_intercept": True}],
        },
    ],
    "rf": [
        {
            "scaler_selection": ["None"],
            "pca_check": [False],
            "model_params": [
                {
                    "n_estimators": 250,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                },
                {
                    "n_estimators": 250,
                    "max_depth": 14,
                    "min_samples_leaf": 2,
                    "max_features": "log2",
                },
            ],
        }
    ],
    "xgb": [
        {
            "scaler_selection": ["None"],
            "pca_check": [False],
            "model_params": [
                {
                    "n_estimators": 250,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
                {
                    "n_estimators": 350,
                    "max_depth": 7,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.8,
                },
            ],
        }
    ],
    "cat": [
        {
            "scaler_selection": ["None"],
            "pca_check": [False],
            "model_params": [
                {
                    "iterations": 300,
                    "depth": 6,
                    "learning_rate": 0.05,
                    "rsm": 0.8,
                    "verbose": False,
                },
                {
                    "iterations": 300,
                    "depth": 8,
                    "learning_rate": 0.1,
                    "rsm": 0.8,
                    "verbose": False,
                },
            ],
        }
    ],
    "lgb": [
        {
            "scaler_selection": ["None"],
            "pca_check": [False],
            "model_params": [
                {
                    "n_estimators": 300,
                    "max_depth": -1,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
                {
                    "n_estimators": 300,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "num_leaves": 63,
                    "subsample": 1.0,
                    "colsample_bytree": 0.8,
                },
            ],
        }
    ],
}


@dataclass(frozen=True)
class ModelComparisonCandidate:
    model_key: str
    scaler_selection: str
    pca_check: bool
    n_components: int
    model_params: dict[str, object]


def compare_model_candidates(
    prepared_data,
    *,
    model_keys: list[str],
    target_transform: str,
    cv_folds: int,
    max_candidates_per_model: int = 4,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_key in model_keys:
        candidates = list(
            islice(
                iter_model_candidates(model_key),
                max_candidates_per_model,
            )
        )
        for candidate in candidates:
            candidate_data = rebuild_prepared_data_for_model(prepared_data, candidate.model_key)
            candidate_components = min(
                candidate.n_components,
                max(1, candidate_data.x_train.shape[1]),
            )
            cv_summary = run_cross_validation(
                candidate_data,
                model_selection=candidate.model_key,
                scaler_selection=candidate.scaler_selection,
                pca_check=candidate.pca_check,
                n_components=candidate_components,
                target_transform=target_transform,
                cv_folds=cv_folds,
                model_params=candidate.model_params,
            )
            rows.append(
                {
                    "Candidate Label": (
                        f"{MODEL_LABELS[candidate.model_key]} | {candidate.scaler_selection} | "
                        f"PCA {'an' if candidate.pca_check else 'aus'}"
                    ),
                    "Model Key": candidate.model_key,
                    "Model": MODEL_LABELS[candidate.model_key],
                    "Scaler": candidate.scaler_selection,
                    "PCA": candidate.pca_check,
                    "PCA Components": candidate_components if candidate.pca_check else "N/A",
                    "Parameters": str(candidate.model_params),
                    "Parameters Raw": candidate.model_params,
                    "CV RMSE": cv_summary.mean_rmse,
                    "CV RMSE Std": cv_summary.std_rmse,
                    "CV MAE": cv_summary.mean_mae,
                    "CV MAE Std": cv_summary.std_mae,
                    "CV R2": cv_summary.mean_r2,
                    "CV R2 Std": cv_summary.std_r2,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["CV RMSE", "CV MAE", "CV R2"],
        ascending=[True, True, False],
    )


def iter_model_candidates(model_key: str):
    for grid in MODEL_SEARCH_SPACE[model_key]:
        for candidate in ParameterGrid(grid):
            yield ModelComparisonCandidate(
                model_key=model_key,
                scaler_selection=candidate["scaler_selection"],
                pca_check=candidate["pca_check"],
                n_components=candidate.get("n_components", 10),
                model_params=candidate["model_params"],
            )
