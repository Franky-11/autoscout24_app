from autoscout24.data.io import load_dataset, load_raw_dataset
from autoscout24.data.schema import EXPECTED_COLUMNS


def test_load_raw_dataset_matches_expected_schema():
    df = load_raw_dataset()

    assert tuple(df.columns) == EXPECTED_COLUMNS
    assert not df.empty


def test_load_dataset_removes_missing_values_and_duplicates():
    df = load_dataset()

    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0
