from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

DATA_SNAPSHOT_YEAR = 2021
ENGINEERED_FEATURES = (
    "car_age",
    "mileage_per_year",
    "log_mileage",
    "log_hp",
    "make_model",
    "fuel_gear_combo",
)

REQUIRED_INPUTS_BY_FEATURE = {
    "car_age": ("year",),
    "mileage_per_year": ("mileage", "year"),
    "log_mileage": ("mileage",),
    "log_hp": ("hp",),
    "make_model": ("make", "model"),
    "fuel_gear_combo": ("fuel", "gear"),
}

CATEGORICAL_FEATURES = {
    "make",
    "model",
    "fuel",
    "gear",
    "offerType",
    "make_model",
    "fuel_gear_combo",
}


def get_engineered_feature_names() -> list[str]:
    return list(ENGINEERED_FEATURES)


def apply_feature_engineering(
    df: pd.DataFrame,
    current_year: int = DATA_SNAPSHOT_YEAR,
) -> pd.DataFrame:
    engineered_df = df.copy()

    car_age = current_year - engineered_df["year"]
    engineered_df["car_age"] = car_age
    engineered_df["mileage_per_year"] = engineered_df["mileage"] / car_age.replace(0, 1)
    engineered_df["log_mileage"] = np.log(engineered_df["mileage"] + 1)
    engineered_df["log_hp"] = np.log(engineered_df["hp"] + 1)
    engineered_df["make_model"] = (
        engineered_df["make"].astype(str) + "_" + engineered_df["model"].astype(str)
    )
    engineered_df["fuel_gear_combo"] = (
        engineered_df["fuel"].astype(str) + "_" + engineered_df["gear"].astype(str)
    )

    for column in ("make_model", "fuel_gear_combo"):
        engineered_df[column] = engineered_df[column].astype("category")

    return engineered_df


def split_feature_types(
    df: pd.DataFrame,
    features: Sequence[str],
) -> tuple[list[str], list[str]]:
    categorical_cols = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    numerical_cols = [feature for feature in features if feature not in categorical_cols]
    return categorical_cols, numerical_cols


def build_feature_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    model_selection: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    categorical_cols, numerical_cols = split_feature_types(df, features)
    feature_frame = df[list(features)].copy()

    if model_selection not in ["lgb", "cat"]:
        feature_frame = pd.get_dummies(feature_frame, columns=categorical_cols, drop_first=True)

    return feature_frame, categorical_cols, numerical_cols


def required_prediction_inputs(features: Sequence[str]) -> list[str]:
    inputs: set[str] = set()

    for feature in features:
        if feature in REQUIRED_INPUTS_BY_FEATURE:
            inputs.update(REQUIRED_INPUTS_BY_FEATURE[feature])
        else:
            inputs.add(feature)

    return sorted(inputs)


def build_vehicle_input_frame(
    vehicle_inputs: Mapping[str, object],
    selected_features: Sequence[str],
    training_columns: Sequence[str],
    categorical_columns: Sequence[str],
    model_selection: str,
    current_year: int = DATA_SNAPSHOT_YEAR,
) -> pd.DataFrame:
    raw_vehicle_df = pd.DataFrame([vehicle_inputs])
    engineered_vehicle_df = apply_feature_engineering(raw_vehicle_df, current_year=current_year)
    engineered_row = engineered_vehicle_df.iloc[0]

    if model_selection not in ["lgb", "cat"]:
        aligned_row = {column: False for column in training_columns}

        for feature in selected_features:
            if feature in categorical_columns:
                dummy_column = f"{feature}_{engineered_row[feature]}"
                if dummy_column in aligned_row:
                    aligned_row[dummy_column] = True
                continue

            if feature in aligned_row:
                aligned_row[feature] = engineered_row[feature]

        return pd.DataFrame([aligned_row], columns=training_columns)

    feature_frame, _, _ = build_feature_frame(
        engineered_vehicle_df,
        selected_features,
        model_selection,
    )

    aligned_frame = feature_frame.reindex(columns=training_columns)
    for column in categorical_columns:
        if column in aligned_frame.columns:
            aligned_frame[column] = aligned_frame[column].astype("category")

    return aligned_frame
