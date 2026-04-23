from collections.abc import Iterable

import pandas as pd

EXPECTED_COLUMNS = (
    "mileage",
    "make",
    "model",
    "fuel",
    "gear",
    "offerType",
    "price",
    "hp",
    "year",
)

CATEGORICAL_COLUMNS = ("make", "model", "fuel", "gear", "offerType")
MODELING_CATEGORICAL_COLUMNS = ("make", "model", "fuel", "gear", "offerType")


def validate_vehicle_schema(
    df: pd.DataFrame,
    expected_columns: Iterable[str] = EXPECTED_COLUMNS,
) -> None:
    expected = list(expected_columns)
    missing = sorted(set(expected) - set(df.columns))
    unexpected = sorted(set(df.columns) - set(expected))

    if missing or unexpected:
        raise ValueError(
            "Unexpected vehicle dataset schema. "
            f"Missing columns: {missing or 'none'}. "
            f"Unexpected columns: {unexpected or 'none'}."
        )
