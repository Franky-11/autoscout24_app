import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

from autoscout24.data.io import load_modeling_dataset
from autoscout24.features.engineering import (
    apply_feature_engineering,
    build_feature_frame,
    get_engineered_feature_names,
    split_feature_types,
)
from autoscout24.modeling.evaluation import EvaluationReport, build_evaluation_report

MODEL_LABELS = {
    "lr": "Linear Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "cat": "CatBoost",
    "lgb": "LightGBM",
}

DEFAULT_BASE_FEATURES = ["hp", "fuel", "gear", "make", "mileage", "year"]


@dataclass(frozen=True)
class PreparedTrainingData:
    base_df: pd.DataFrame
    engineered_df: pd.DataFrame
    selected_features: list[str]
    engineered_feature_options: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    feature_frame: pd.DataFrame
    categorical_dummy_columns: list[str]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: list[int]
    test_indices: list[int]


@dataclass(frozen=True)
class MetricSummary:
    rmse: float
    r2: float
    mae: float


@dataclass(frozen=True)
class CrossValidationSummary:
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    std_r2: float
    mean_mae: float
    std_mae: float


@dataclass(frozen=True)
class BaselineSummary:
    name: str
    rmse: float
    r2: float
    mae: float


@dataclass(frozen=True)
class TrainingRunResult:
    model: object
    scaler: object
    pipeline: object
    y_pred: pd.Series
    rmse: float
    r2: float
    mae: float
    train_time: float
    cv_summary: CrossValidationSummary
    baselines: list[BaselineSummary]
    evaluation_report: EvaluationReport


def build_model_and_scaler(
    model_selection: str,
    scaler_selection: str,
    categorical_columns: list[str],
    model_params: dict[str, object] | None = None,
) -> tuple[object, object]:
    params = model_params or {}
    if model_selection == "lr":
        model = LinearRegression(**({"fit_intercept": True} | params))
    elif model_selection == "rf":
        model = RandomForestRegressor(
            **{
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "max_features": "log2",
                "random_state": 42,
            }
            | params
        )
    elif model_selection == "xgb":
        model = XGBRegressor(
            **{
                "random_state": 42,
                "n_jobs": -1,
                "colsample_bytree": 0.8,
                "learning_rate": 0.1,
                "max_depth": 7,
                "n_estimators": 300,
                "subsample": 1.0,
            }
            | params
        )
    elif model_selection == "cat":
        model = CatBoostRegressor(
            **{
                "iterations": 300,
                "learning_rate": 0.1,
                "depth": 7,
                "rsm": 0.8,
                "subsample": 1.0,
                "random_seed": 42,
                "loss_function": "RMSE",
                "verbose": False,
                "cat_features": categorical_columns,
            }
            | params
        )
    elif model_selection == "lgb":
        model = LGBMRegressor(
            **{
                "random_state": 42,
                "n_jobs": -1,
                "colsample_bytree": 0.8,
                "learning_rate": 0.1,
                "max_depth": 7,
                "n_estimators": 300,
                "subsample": 1.0,
            }
            | params
        )
    else:
        raise KeyError(f"Unknown model selection: {model_selection}")

    scaler_dict = {
        "Standard": StandardScaler(),
        "MinMax": MinMaxScaler(),
        "None": "passthrough",
    }
    return model, scaler_dict[scaler_selection]


def build_pipeline(
    pca_check: bool,
    n_components: int,
    model_selection: str,
    model: object,
    scaler: object,
    categorical_dummy_columns: list[str],
    numeric_columns: list[str],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_columns),
            ("cat", "passthrough", categorical_dummy_columns),
        ]
    )

    pipeline_steps: list[tuple[str, object]] = [("preprocessor", preprocessor)]
    if pca_check:
        pipeline_steps.append(("pca", PCA(n_components=n_components)))
    pipeline_steps.append((model_selection, model))
    pipeline = Pipeline(pipeline_steps)

    if model_selection in ["cat", "lgb"]:
        pipeline = Pipeline([(model_selection, model)])

    return pipeline


def calculate_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[float, float, float]:
    rmse = metrics.root_mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    return rmse, r2, mae


def _transform_target(y: pd.Series, target_transform: str) -> pd.Series:
    if target_transform == "log1p":
        return np.log1p(y)
    return y


def _inverse_target(y_pred: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log1p":
        return np.expm1(np.clip(y_pred, a_min=-20, a_max=20))
    return y_pred


def predict_with_pipeline(
    pipeline: object,
    feature_frame: pd.DataFrame,
    target_transform: str,
) -> np.ndarray:
    raw_predictions = pipeline.predict(feature_frame)
    return _inverse_target(raw_predictions, target_transform)


def calculate_pca_explained_variance(
    x_train: pd.DataFrame,
    model_selection: str,
    scaler_selection: str,
    categorical_dummy_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.Series | None:
    _, scaler = build_model_and_scaler(
        model_selection,
        scaler_selection,
        categorical_columns,
    )

    if model_selection in ["lgb", "cat"]:
        return None

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_columns),
            ("cat", "passthrough", categorical_dummy_columns),
        ]
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("pca", PCA(n_components=x_train.shape[1]))]
    )
    pipeline.fit_transform(x_train)
    return pipeline["pca"].explained_variance_ratio_.cumsum()


def prepare_training_data(
    base_features: list[str],
    engineered_features: list[str],
    model_selection: str,
    test_size: float,
) -> PreparedTrainingData:
    base_df = load_modeling_dataset()
    engineered_df = apply_feature_engineering(base_df)
    selected_features = [*base_features, *engineered_features]
    categorical_columns, numeric_columns = split_feature_types(engineered_df, selected_features)
    feature_frame, _, _ = build_feature_frame(engineered_df, selected_features, model_selection)
    categorical_dummy_columns = [
        column for column in feature_frame.columns if column not in numeric_columns
    ]
    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        engineered_df["price"],
        test_size=test_size,
        random_state=42,
    )

    return PreparedTrainingData(
        base_df=base_df,
        engineered_df=engineered_df,
        selected_features=selected_features,
        engineered_feature_options=get_engineered_feature_names(),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        feature_frame=feature_frame,
        categorical_dummy_columns=categorical_dummy_columns,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=x_train.index.tolist(),
        test_indices=x_test.index.tolist(),
    )


def rebuild_prepared_data_for_model(
    prepared_data: PreparedTrainingData,
    model_selection: str,
) -> PreparedTrainingData:
    categorical_columns, numeric_columns = split_feature_types(
        prepared_data.engineered_df,
        prepared_data.selected_features,
    )
    feature_frame, _, _ = build_feature_frame(
        prepared_data.engineered_df,
        prepared_data.selected_features,
        model_selection,
    )
    categorical_dummy_columns = [
        column for column in feature_frame.columns if column not in numeric_columns
    ]

    return PreparedTrainingData(
        base_df=prepared_data.base_df,
        engineered_df=prepared_data.engineered_df,
        selected_features=prepared_data.selected_features,
        engineered_feature_options=prepared_data.engineered_feature_options,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        feature_frame=feature_frame,
        categorical_dummy_columns=categorical_dummy_columns,
        x_train=feature_frame.loc[prepared_data.train_indices],
        x_test=feature_frame.loc[prepared_data.test_indices],
        y_train=prepared_data.y_train,
        y_test=prepared_data.y_test,
        train_indices=prepared_data.train_indices,
        test_indices=prepared_data.test_indices,
    )


def train_model_run(
    prepared_data: PreparedTrainingData,
    model_selection: str,
    scaler_selection: str,
    pca_check: bool,
    n_components: int,
    target_transform: str = "raw",
    cv_folds: int = 3,
    model_params: dict[str, object] | None = None,
) -> TrainingRunResult:
    model, scaler = build_model_and_scaler(
        model_selection,
        scaler_selection,
        prepared_data.categorical_columns,
        model_params=model_params,
    )
    pipeline = build_pipeline(
        pca_check,
        n_components,
        model_selection,
        model,
        scaler,
        prepared_data.categorical_dummy_columns,
        prepared_data.numeric_columns,
    )

    start = time.perf_counter()
    y_train_transformed = _transform_target(prepared_data.y_train, target_transform)
    if model_selection == "lgb":
        pipeline.fit(
            prepared_data.x_train,
            y_train_transformed,
            lgb__categorical_feature=prepared_data.categorical_columns,
        )
    else:
        pipeline.fit(prepared_data.x_train, y_train_transformed)
    train_time = time.perf_counter() - start

    y_pred = _inverse_target(pipeline.predict(prepared_data.x_test), target_transform)
    rmse, r2, mae = calculate_regression_metrics(prepared_data.y_test, y_pred)
    cv_summary = run_cross_validation(
        prepared_data,
        model_selection=model_selection,
        scaler_selection=scaler_selection,
        pca_check=pca_check,
        n_components=n_components,
        target_transform=target_transform,
        cv_folds=cv_folds,
        model_params=model_params,
    )
    baselines = evaluate_holdout_baselines(prepared_data)
    evaluation_report = build_evaluation_report(
        prepared_data.y_test,
        y_pred,
        prepared_data.engineered_df,
    )

    return TrainingRunResult(
        model=model,
        scaler=scaler,
        pipeline=pipeline,
        y_pred=y_pred,
        rmse=rmse,
        r2=r2,
        mae=mae,
        train_time=train_time,
        cv_summary=cv_summary,
        baselines=baselines,
        evaluation_report=evaluation_report,
    )


def run_cross_validation(
    prepared_data: PreparedTrainingData,
    model_selection: str,
    scaler_selection: str,
    pca_check: bool,
    n_components: int,
    target_transform: str,
    cv_folds: int,
    model_params: dict[str, object] | None = None,
) -> CrossValidationSummary:
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rmse_scores: list[float] = []
    r2_scores: list[float] = []
    mae_scores: list[float] = []

    for train_idx, val_idx in splitter.split(prepared_data.feature_frame):
        x_train = prepared_data.feature_frame.iloc[train_idx]
        x_val = prepared_data.feature_frame.iloc[val_idx]
        y_train = prepared_data.engineered_df["price"].iloc[train_idx]
        y_val = prepared_data.engineered_df["price"].iloc[val_idx]

        model, scaler = build_model_and_scaler(
            model_selection,
            scaler_selection,
            prepared_data.categorical_columns,
            model_params=model_params,
        )
        pipeline = build_pipeline(
            pca_check,
            n_components,
            model_selection,
            model,
            scaler,
            prepared_data.categorical_dummy_columns,
            prepared_data.numeric_columns,
        )
        y_train_transformed = _transform_target(y_train, target_transform)
        if model_selection == "lgb":
            pipeline.fit(
                x_train,
                y_train_transformed,
                lgb__categorical_feature=prepared_data.categorical_columns,
            )
        else:
            pipeline.fit(x_train, y_train_transformed)

        y_pred = _inverse_target(pipeline.predict(x_val), target_transform)
        rmse, r2, mae = calculate_regression_metrics(y_val, y_pred)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mae_scores.append(mae)

    return CrossValidationSummary(
        mean_rmse=float(np.mean(rmse_scores)),
        std_rmse=float(np.std(rmse_scores)),
        mean_r2=float(np.mean(r2_scores)),
        std_r2=float(np.std(r2_scores)),
        mean_mae=float(np.mean(mae_scores)),
        std_mae=float(np.std(mae_scores)),
    )


def evaluate_holdout_baselines(prepared_data: PreparedTrainingData) -> list[BaselineSummary]:
    train_df = prepared_data.base_df.loc[prepared_data.train_indices].copy()
    test_df = prepared_data.base_df.loc[prepared_data.test_indices].copy()

    global_median = train_df["price"].median()
    global_predictions = np.full(shape=len(test_df), fill_value=global_median)

    make_medians = train_df.groupby("make", observed=False)["price"].median().to_dict()
    make_predictions = (
        test_df["make"].astype(str).map(make_medians).astype(float).fillna(global_median).to_numpy()
    )

    baselines = []
    for name, predictions in [
        ("Global Median", global_predictions),
        ("Make Median", make_predictions),
    ]:
        rmse, r2, mae = calculate_regression_metrics(test_df["price"], predictions)
        baselines.append(BaselineSummary(name=name, rmse=rmse, r2=r2, mae=mae))

    return baselines
