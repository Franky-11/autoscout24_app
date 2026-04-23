import pandas as pd

from autoscout24.data.schema import MODELING_CATEGORICAL_COLUMNS


def drop_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().dropna().copy()


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


def prepare_modeling_dataset(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    cleaned = drop_missing_and_duplicates(df)
    _, filtered, _ = remove_outliers(cleaned, factor=factor)
    modeling_df = filtered.drop(columns=["price_u_limit", "mileage_u_limit", "hp_u_limit"]).copy()

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
