from autoscout24.data.cleaning import get_outlier_counts, prepare_modeling_dataset, remove_outliers
from autoscout24.data.io import load_raw_dataset


def test_prepare_modeling_dataset_drops_offer_type_and_sets_categories():
    df = prepare_modeling_dataset(load_raw_dataset())

    assert "offerType" not in df.columns
    assert str(df["make"].dtype) == "category"
    assert str(df["model"].dtype) == "category"
    assert str(df["fuel"].dtype) == "category"
    assert str(df["gear"].dtype) == "category"


def test_remove_outliers_returns_filtered_dataset_and_outlier_counts():
    raw_df = load_raw_dataset().drop_duplicates().dropna()
    merged_df, filtered_df, _ = remove_outliers(raw_df, factor=1.5)
    outlier_counts = get_outlier_counts(merged_df, filtered_df)

    assert len(filtered_df) <= len(merged_df)
    assert {"year", "outliers"} == set(outlier_counts.columns)
