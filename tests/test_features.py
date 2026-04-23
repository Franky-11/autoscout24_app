from autoscout24.data.io import load_modeling_dataset
from autoscout24.features.engineering import (
    apply_feature_engineering,
    build_feature_frame,
    build_vehicle_input_frame,
    get_engineered_feature_names,
    required_prediction_inputs,
    split_feature_types,
)


def test_feature_engineering_adds_expected_columns():
    df = load_modeling_dataset().head(5)
    engineered_df = apply_feature_engineering(df)

    for feature in get_engineered_feature_names():
        assert feature in engineered_df.columns


def test_required_prediction_inputs_expand_derived_features():
    inputs = required_prediction_inputs(
        ["hp", "make_model", "fuel_gear_combo", "log_mileage", "offerType"]
    )

    assert inputs == ["fuel", "gear", "hp", "make", "mileage", "model", "offerType"]


def test_single_vehicle_frame_matches_training_transformation_for_one_hot_models():
    base_df = load_modeling_dataset().head(20).copy()
    selected_features = [
        "hp",
        "fuel",
        "gear",
        "make",
        "model",
        "mileage",
        "year",
        "car_age",
        "mileage_per_year",
        "log_mileage",
        "log_hp",
        "make_model",
        "fuel_gear_combo",
    ]

    engineered_df = apply_feature_engineering(base_df)
    training_frame, categorical_cols, _ = build_feature_frame(
        engineered_df,
        selected_features,
        model_selection="rf",
    )
    first_vehicle = base_df.iloc[0]

    prediction_frame = build_vehicle_input_frame(
        vehicle_inputs={
            "hp": first_vehicle["hp"],
            "year": first_vehicle["year"],
            "mileage": first_vehicle["mileage"],
            "make": first_vehicle["make"],
            "fuel": first_vehicle["fuel"],
            "gear": first_vehicle["gear"],
            "model": first_vehicle["model"],
        },
        selected_features=selected_features,
        training_columns=training_frame.columns.tolist(),
        categorical_columns=categorical_cols,
        model_selection="rf",
    )

    assert prediction_frame.columns.tolist() == training_frame.columns.tolist()
    assert prediction_frame.iloc[0].to_dict() == training_frame.iloc[0].to_dict()


def test_single_vehicle_frame_preserves_categories_for_catboost_like_models():
    base_df = load_modeling_dataset().head(5).copy()
    selected_features = ["hp", "fuel", "gear", "make", "model", "car_age", "make_model"]

    engineered_df = apply_feature_engineering(base_df)
    training_frame, categorical_cols, _ = build_feature_frame(
        engineered_df,
        selected_features,
        model_selection="cat",
    )
    first_vehicle = base_df.iloc[0]

    prediction_frame = build_vehicle_input_frame(
        vehicle_inputs={
            "hp": first_vehicle["hp"],
            "year": first_vehicle["year"],
            "mileage": first_vehicle["mileage"],
            "make": first_vehicle["make"],
            "fuel": first_vehicle["fuel"],
            "gear": first_vehicle["gear"],
            "model": first_vehicle["model"],
        },
        selected_features=selected_features,
        training_columns=training_frame.columns.tolist(),
        categorical_columns=categorical_cols,
        model_selection="cat",
    )

    inferred_categorical_cols, inferred_numeric_cols = split_feature_types(
        engineered_df,
        selected_features,
    )

    assert inferred_categorical_cols == categorical_cols
    assert inferred_numeric_cols == ["hp", "car_age"]
    assert str(prediction_frame["make"].dtype) == "category"
    assert str(prediction_frame["make_model"].dtype) == "category"
