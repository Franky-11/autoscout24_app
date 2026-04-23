import streamlit as st

from autoscout24.exploration.charts import (
    build_category_share_chart,
    build_relation_chart,
    build_top_sales_chart,
)
from autoscout24.exploration.service import (
    build_category_share_frame,
    build_category_volume_frame,
    build_top_sales_frame,
    filter_relation_dataset,
    load_clean_exploration_dataset,
)

DEFAULT_CATEGORY = "make"
DEFAULT_YEARS = [2011, 2021]
DEFAULT_TOP_YEARS = [2011]
DEFAULT_MAKES = ["Audi", "Opel"]
DEFAULT_FUELS = ["Gasoline", "Diesel"]
DEFAULT_GEAR = ["Manual"]


@st.cache_data
def _load_dashboard_dataset():
    return load_clean_exploration_dataset()


def _render_category_share_tab(df, category: str, years: list[int]) -> None:
    columns = st.columns(len(years))
    for index, year in enumerate(years):
        counts_df = build_category_share_frame(df, category, year)
        total_sales = int(df[df["year"] == year]["make"].size)
        with columns[index]:
            st.markdown(f"**Verkäufe Erstzulassung {year}  \n{total_sales}**")
            st.plotly_chart(
                build_category_share_chart(counts_df, category, "count", "Kategorie"),
                key=f"pie_sales_{category}_{year}",
            )


def _render_category_volume_tab(df, category: str, years: list[int]) -> None:
    columns = st.columns(len(years))
    for index, year in enumerate(years):
        volume_df = build_category_volume_frame(df, category, year)
        total_volume = df[df["year"] == year]["price"].sum() / 1_000_000
        with columns[index]:
            st.markdown(f"**Preis Gesamt Erstzulassung {year}  \n{total_volume:.2f} Mio.**")
            st.plotly_chart(
                build_category_share_chart(volume_df, category, "sales_volume", "Kategorie"),
                key=f"pie_volume_{category}_{year}",
            )


def render_page() -> None:
    df = _load_dashboard_dataset()

    st.title(":material/dashboard: Dashboard")
    col1, col2 = st.columns(2)
    height = 650

    with col1:
        with st.container(border=True, height=height):
            col_cat, col_year = st.columns(2)
            with col_cat:
                with st.expander("Kategorie wählen"):
                    category = st.segmented_control(
                        "Kategorie",
                        options=["make", "fuel", "gear", "model", "offerType"],
                        default=DEFAULT_CATEGORY,
                    )
            with col_year:
                with st.expander("Erstzulassung wählen"):
                    years = st.multiselect(
                        "Erstzulassung wählen:",
                        options=list(range(2011, 2022)),
                        default=DEFAULT_YEARS,
                    )

            tab_sales, tab_sales_volume = st.tabs(["Verkäufe", "Preis"])
            with tab_sales:
                _render_category_share_tab(df, category, years)
            with tab_sales_volume:
                _render_category_volume_tab(df, category, years)

    with col2:
        with st.container(border=True, height=height):
            col_mark, col_year, col_gear, col_fuel = st.columns(4)
            with col_mark:
                st.markdown("**Top 10 Marken**")
            with col_year:
                top_years = st.multiselect(
                    "Erstzulassung",
                    options=[*sorted(df["year"].unique().tolist()), "alle"],
                    default=DEFAULT_TOP_YEARS,
                    key="year",
                )
            with col_gear:
                gear = st.multiselect(
                    "Schaltung",
                    options=sorted(df["gear"].unique().tolist()),
                    default=DEFAULT_GEAR,
                    key="gear",
                )
            with col_fuel:
                fuel = st.multiselect(
                    "Kraftstoff",
                    options=sorted(df["fuel"].unique().tolist()),
                    default=["Gasoline"],
                    key="fuel",
                )

            selected_years = None if "alle" in top_years else [int(year) for year in top_years]
            top10 = build_top_sales_frame(df, fuel, gear, selected_years)
            st.plotly_chart(
                build_top_sales_chart(top10, include_year_dimension=selected_years is not None)
            )

    with st.container(border=True, height=900):
        cols = st.columns(3)
        with cols[0]:
            fuel_price = st.multiselect(
                "Kraftstoff",
                options=sorted(df["fuel"].unique().tolist()),
                default=DEFAULT_FUELS,
            )
        with cols[1]:
            make = st.multiselect(
                "Marke",
                options=sorted(df["make"].unique().tolist() + ["Alle"]),
                default=DEFAULT_MAKES,
            )
        with cols[2]:
            gear_price = st.segmented_control(
                "Schaltung",
                options=sorted(df["gear"].unique().tolist()),
                default=DEFAULT_GEAR,
                selection_mode="multi",
            )

        relation_df, color_value = filter_relation_dataset(df, fuel_price, gear_price, make)
        tabs = st.tabs(
            [
                "Mileage vs. Erstzulassung",
                "Preis vs. Erstzulassung",
                "Preis vs. Mileage",
            ]
        )
        with tabs[1]:
            st.markdown("**Preis vs. Erstzulassung**")
            st.plotly_chart(
                build_relation_chart(relation_df, color_value, mode="price_year"),
                use_container_width=True,
            )
        with tabs[0]:
            st.markdown("**Mileage vs. Erstzulassung**")
            st.plotly_chart(
                build_relation_chart(relation_df, color_value, mode="mileage_year"),
                use_container_width=True,
            )
        with tabs[2]:
            st.markdown("**Preis vs. Mileage**")
            st.plotly_chart(
                build_relation_chart(relation_df, color_value, mode="price_mileage"),
                use_container_width=True,
            )
