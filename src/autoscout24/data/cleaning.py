from datetime import datetime

import pandas as pd

from autoscout24.data.schema import MODELING_CATEGORICAL_COLUMNS

MIN_VALID_YEAR = 1950
MAX_VALID_PRICE = 2_000_000
MAX_VALID_MILEAGE = 2_000_000
MAX_VALID_HP = 2_000


def drop_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().dropna().copy()


def apply_plausibility_filters(df: pd.DataFrame) -> pd.DataFrame:
    max_valid_year = datetime.now().year + 1
    valid_rows = (
        df["price"].gt(0)
        & df["price"].le(MAX_VALID_PRICE)
        & df["mileage"].ge(0)
        & df["mileage"].le(MAX_VALID_MILEAGE)
        & df["hp"].gt(0)
        & df["hp"].le(MAX_VALID_HP)
        & df["year"].between(MIN_VALID_YEAR, max_valid_year)
    )
    return df.loc[valid_rows].copy()


def calculate_iqr_bounds(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    bounds = (
        df.groupby("year")
        .agg(
            price_q1=("price", lambda values: values.quantile(0.25)),
            price_q3=("price", lambda values: values.quantile(0.75)),
            mileage_q1=("mileage", lambda values: values.quantile(0.25)),
            mileage_q3=("mileage", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    bounds["price_u_limit"] = (
        bounds["price_q3"] + (bounds["price_q3"] - bounds["price_q1"]) * factor
    )
    bounds["mileage_u_limit"] = (
        bounds["mileage_q3"] + (bounds["mileage_q3"] - bounds["mileage_q1"]) * factor
    )
    return bounds


def remove_outliers(
    df: pd.DataFrame,
    factor: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    iqr_bounds = calculate_iqr_bounds(df, factor=factor)
    df_merged = pd.merge(
        df,
        iqr_bounds[["year", "price_u_limit", "mileage_u_limit"]],
        on="year",
        how="left",
    )

    hp_iqr_bounds = df_merged["hp"].quantile([0.25, 0.75])
    hp_u_limit = hp_iqr_bounds[0.75] + (hp_iqr_bounds[0.75] - hp_iqr_bounds[0.25]) * factor
    df_merged["hp_u_limit"] = hp_u_limit

    filtered = df_merged[
        (df_merged["price"] <= df_merged["price_u_limit"])
        & (df_merged["mileage"] <= df_merged["mileage_u_limit"])
        & (df_merged["hp"] <= df_merged["hp_u_limit"])
    ].copy()

    return df_merged, filtered, iqr_bounds


def prepare_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = drop_missing_and_duplicates(df)
    modeling_df = apply_plausibility_filters(cleaned)

    for column in MODELING_CATEGORICAL_COLUMNS:
        modeling_df[column] = modeling_df[column].astype("category")

    return modeling_df


def get_outlier_counts(df_merged: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    outliers = df_merged.groupby("year").size() - filtered_df.groupby("year").size()
    outlier_df = outliers.reset_index()
    outlier_df.columns = ["year", "outliers"]
    outlier_df = outlier_df.sort_values(by="outliers", ascending=False)
    outlier_df["year"] = pd.Categorical(
        outlier_df["year"],
        categories=outlier_df["year"].tolist(),
        ordered=True,
    )
    return outlier_df
