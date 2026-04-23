from sklearn.model_selection import train_test_split

from autoscout24.data.io import load_modeling_dataset
from autoscout24.features.engineering import (
    apply_feature_engineering,
    build_feature_frame,
    split_feature_types,
)
from autoscout24.modeling.service import (
    PreparedTrainingData,
    predict_with_pipeline,
    prepare_training_data,
    train_model_run,
)


def test_prepare_training_data_builds_consistent_bundle():
    prepared_data = prepare_training_data(
        base_features=["hp", "fuel", "gear", "make", "mileage", "year", "offerType"],
        engineered_features=["car_age", "log_mileage"],
        model_selection="rf",
        test_size=0.2,
    )

    assert prepared_data.selected_features == [
        "hp",
        "fuel",
        "gear",
        "make",
        "mileage",
        "year",
        "offerType",
        "car_age",
        "log_mileage",
    ]
    assert len(prepared_data.feature_frame) == len(prepared_data.engineered_df)
    assert len(prepared_data.x_train) + len(prepared_data.x_test) == len(
        prepared_data.feature_frame
    )
    assert set(prepared_data.categorical_columns) == {"fuel", "gear", "make", "offerType"}


def test_train_model_run_returns_metrics_and_predictions():
    base_df = load_modeling_dataset().head(250).copy()
    engineered_df = apply_feature_engineering(base_df)
    selected_features = ["hp", "mileage", "year", "car_age", "log_mileage"]
    categorical_columns, numeric_columns = split_feature_types(engineered_df, selected_features)
    feature_frame, _, _ = build_feature_frame(
        engineered_df,
        selected_features,
        model_selection="lr",
    )
    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        engineered_df["price"],
        test_size=0.2,
        random_state=42,
    )

    prepared_data = PreparedTrainingData(
        base_df=base_df,
        engineered_df=engineered_df,
        selected_features=selected_features,
        engineered_feature_options=["car_age", "log_mileage"],
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        feature_frame=feature_frame,
        categorical_dummy_columns=[
            column for column in feature_frame.columns if column not in numeric_columns
        ],
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=x_train.index.tolist(),
        test_indices=x_test.index.tolist(),
    )

    run_result = train_model_run(
        prepared_data=prepared_data,
        model_selection="lr",
        scaler_selection="Standard",
        pca_check=False,
        n_components=2,
    )

    assert len(run_result.y_pred) == len(y_test)
    assert run_result.train_time >= 0
    assert isinstance(run_result.rmse, float)
    assert isinstance(run_result.r2, float)
    assert isinstance(run_result.mae, float)
    assert len(run_result.baselines) == 2
    assert run_result.cv_summary.mean_rmse > 0


def test_log_target_predictions_are_inverted_back_to_price_scale():
    prepared_data = prepare_training_data(
        base_features=["hp", "mileage", "year", "make"],
        engineered_features=["car_age", "log_mileage"],
        model_selection="lr",
        test_size=0.2,
    )
    run_result = train_model_run(
        prepared_data=prepared_data,
        model_selection="lr",
        scaler_selection="Standard",
        pca_check=False,
        n_components=2,
        target_transform="log1p",
        cv_folds=3,
    )

    predictions = predict_with_pipeline(
        run_result.pipeline,
        prepared_data.x_test.head(3),
        target_transform="log1p",
    )

    assert len(predictions) == 3
    assert (predictions > 0).all()
