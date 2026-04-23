import pandas as pd

from autoscout24.data.cleaning import (
    apply_plausibility_filters,
    get_outlier_counts,
    prepare_modeling_dataset,
    remove_outliers,
)
from autoscout24.data.io import load_raw_dataset


def test_prepare_modeling_dataset_keeps_offer_type_and_sets_categories():
    df = prepare_modeling_dataset(load_raw_dataset())

    assert str(df["make"].dtype) == "category"
    assert str(df["model"].dtype) == "category"
    assert str(df["fuel"].dtype) == "category"
    assert str(df["gear"].dtype) == "category"
    assert str(df["offerType"].dtype) == "category"


def test_remove_outliers_returns_filtered_dataset_and_outlier_counts():
    raw_df = load_raw_dataset().drop_duplicates().dropna()
    merged_df, filtered_df, _ = remove_outliers(raw_df, factor=1.5)
    outlier_counts = get_outlier_counts(merged_df, filtered_df)

    assert len(filtered_df) <= len(merged_df)
    assert {"year", "outliers"} == set(outlier_counts.columns)


def test_apply_plausibility_filters_removes_obviously_invalid_rows():
    df = pd.DataFrame(
        [
            {
                "price": 25_000,
                "mileage": 80_000,
                "hp": 150,
                "year": 2019,
            },
            {
                "price": -1,
                "mileage": 80_000,
                "hp": 150,
                "year": 2019,
            },
            {
                "price": 25_000,
                "mileage": -50,
                "hp": 150,
                "year": 2019,
            },
            {
                "price": 25_000,
                "mileage": 80_000,
                "hp": 0,
                "year": 2019,
            },
            {
                "price": 25_000,
                "mileage": 80_000,
                "hp": 150,
                "year": 1900,
            },
        ]
    )

    filtered_df = apply_plausibility_filters(df)

    assert len(filtered_df) == 1
    assert filtered_df.iloc[0].to_dict() == {
        "price": 25_000,
        "mileage": 80_000,
        "hp": 150,
        "year": 2019,
    }
