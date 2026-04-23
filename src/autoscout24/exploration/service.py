from dataclasses import dataclass

import pandas as pd

from autoscout24.data.cleaning import get_outlier_counts, remove_outliers
from autoscout24.data.io import load_dataset, load_raw_dataset


@dataclass(frozen=True)
class DatasetOverview:
    raw_df: pd.DataFrame
    null_counts: pd.Series
    total_missing: int
    duplicate_count: int


@dataclass(frozen=True)
class OutlierAnalysis:
    merged_df: pd.DataFrame
    filtered_df: pd.DataFrame
    bounds_df: pd.DataFrame
    outlier_counts: pd.DataFrame


def load_clean_exploration_dataset() -> pd.DataFrame:
    return load_dataset()


def load_dataset_overview() -> DatasetOverview:
    raw_df = load_raw_dataset()
    null_counts = raw_df.isnull().sum().sort_values(ascending=False)
    return DatasetOverview(
        raw_df=raw_df,
        null_counts=null_counts,
        total_missing=int(null_counts.sum()),
        duplicate_count=int(raw_df.duplicated().sum()),
    )


def analyze_outliers(df: pd.DataFrame, factor: float = 1.5) -> OutlierAnalysis:
    merged_df, filtered_df, bounds_df = remove_outliers(df, factor=factor)
    return OutlierAnalysis(
        merged_df=merged_df,
        filtered_df=filtered_df,
        bounds_df=bounds_df,
        outlier_counts=get_outlier_counts(merged_df, filtered_df),
    )


def build_category_share_frame(
    df: pd.DataFrame,
    category: str,
    year: int,
) -> pd.DataFrame:
    return (
        df[df["year"] == year]
        .groupby([category, "year"], observed=False)
        .size()
        .reset_index(name="count")
    )


def build_category_volume_frame(
    df: pd.DataFrame,
    category: str,
    year: int,
) -> pd.DataFrame:
    return (
        df[df["year"] == year]
        .groupby([category, "year"], observed=False)["price"]
        .sum()
        .reset_index(name="sales_volume")
    )


def build_top_sales_frame(
    df: pd.DataFrame,
    fuels: list[str],
    gears: list[str],
    years: list[int] | None,
) -> pd.DataFrame:
    base_filtered = df[(df["fuel"].isin(fuels)) & (df["gear"].isin(gears))].copy()

    if years:
        filtered = base_filtered[base_filtered["year"].isin(years)]
        counted = (
            filtered.groupby(["year", "make", "fuel", "gear"], observed=False)
            .size()
            .reset_index(name="count")
        )
        return (
            counted.sort_values(
                by=["year", "fuel", "gear", "count", "make"],
                ascending=[True, True, True, False, True],
            )
            .groupby(["year", "fuel", "gear"], observed=False, group_keys=False)
            .head(10)
            .reset_index(drop=True)
        )

    counted = (
        base_filtered.groupby(["make", "fuel", "gear"], observed=False)
        .size()
        .reset_index(name="count")
    )
    return (
        counted.sort_values(
            by=["fuel", "gear", "count", "make"],
            ascending=[True, True, False, True],
        )
        .groupby(["fuel", "gear"], observed=False, group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )


def filter_relation_dataset(
    df: pd.DataFrame,
    fuels: list[str],
    gears: list[str],
    makes: list[str],
) -> tuple[pd.DataFrame, str | None]:
    filtered_df = analyze_outliers(df, factor=1.5).filtered_df
    filtered_df = filtered_df[
        (filtered_df["fuel"].isin(fuels)) & (filtered_df["gear"].isin(gears))
    ].copy()

    if "Alle" in makes:
        return filtered_df, None

    return filtered_df[filtered_df["make"].isin(makes)].copy(), "make"
