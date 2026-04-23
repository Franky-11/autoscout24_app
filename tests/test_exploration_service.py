from autoscout24.exploration.service import (
    analyze_outliers,
    build_category_share_frame,
    build_top_sales_frame,
    filter_relation_dataset,
    load_clean_exploration_dataset,
    load_dataset_overview,
)


def test_load_dataset_overview_reports_missing_and_duplicates():
    overview = load_dataset_overview()

    assert not overview.raw_df.empty
    assert overview.total_missing == int(overview.raw_df.isnull().sum().sum())
    assert overview.duplicate_count == int(overview.raw_df.duplicated().sum())


def test_analyze_outliers_returns_counts_and_bounds():
    df = load_clean_exploration_dataset()
    analysis = analyze_outliers(df, factor=1.5)

    assert len(analysis.filtered_df) <= len(analysis.merged_df)
    assert {"year", "price_u_limit", "mileage_u_limit"} <= set(analysis.bounds_df.columns)
    assert {"year", "outliers"} == set(analysis.outlier_counts.columns)


def test_build_top_sales_frame_supports_global_and_year_specific_modes():
    df = load_clean_exploration_dataset()

    global_top = build_top_sales_frame(df, ["Gasoline"], ["Manual"], years=None)
    yearly_top = build_top_sales_frame(df, ["Gasoline"], ["Manual"], years=[2018])

    assert {"make", "fuel", "gear", "count"} <= set(global_top.columns)
    assert "year" not in global_top.columns
    assert {"year", "make", "fuel", "gear", "count"} <= set(yearly_top.columns)
    assert set(yearly_top["year"].unique()) == {2018}


def test_filter_relation_dataset_keeps_make_filter_optional():
    df = load_clean_exploration_dataset()

    filtered_all, color_all = filter_relation_dataset(df, ["Gasoline"], ["Manual"], ["Alle"])
    filtered_make, color_make = filter_relation_dataset(df, ["Gasoline"], ["Manual"], ["Audi"])

    assert color_all is None
    assert not filtered_all.empty
    assert color_make == "make"
    assert set(filtered_make["make"].unique()) == {"Audi"}


def test_build_category_share_frame_aggregates_single_year_counts():
    df = load_clean_exploration_dataset()
    frame = build_category_share_frame(df, "make", 2018)

    assert not frame.empty
    assert set(frame["year"].unique()) == {2018}
