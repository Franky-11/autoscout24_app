from autoscout24.modeling.service import prepare_training_data
from autoscout24.modeling.tuning import compare_model_candidates


def test_compare_model_candidates_rebuilds_features_per_model():
    prepared_data = prepare_training_data(
        base_features=["hp", "fuel", "gear", "make", "mileage", "year", "offerType"],
        engineered_features=["car_age", "log_mileage"],
        model_selection="rf",
        test_size=0.2,
    )

    comparison_df = compare_model_candidates(
        prepared_data,
        model_keys=["lr", "cat"],
        target_transform="raw",
        cv_folds=3,
        max_candidates_per_model=1,
    )

    assert not comparison_df.empty
    assert set(comparison_df["Model Key"]) == {"lr", "cat"}
    assert {"CV RMSE", "CV MAE", "CV R2"} <= set(comparison_df.columns)
