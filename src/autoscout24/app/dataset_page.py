import streamlit as st

from autoscout24.exploration.charts import (
    build_category_count_chart,
    build_distribution_chart,
    build_null_counts_chart,
    build_outlier_chart,
    build_stage_summary_chart,
)
from autoscout24.exploration.service import (
    analyze_outliers,
    build_missing_summary,
    build_numeric_summary,
    build_stage_summary,
    load_dataset_overview,
)


@st.cache_data
def _load_dataset_overview_cached(cache_version: int = 2):
    return load_dataset_overview()


def _load_dataset_overview():
    overview = _load_dataset_overview_cached()
    required_fields = (
        "raw_df",
        "clean_df",
        "modeling_df",
        "null_counts",
        "total_missing",
        "duplicate_count",
    )
    if all(hasattr(overview, field) for field in required_fields):
        return overview

    _load_dataset_overview_cached.clear()
    return _load_dataset_overview_cached()


def render_page() -> None:
    overview = _load_dataset_overview()
    stage_summary = build_stage_summary(overview)
    missing_summary = build_missing_summary(overview.raw_df)
    numeric_summary = build_numeric_summary(overview.raw_df, ["price", "mileage", "hp", "year"])

    st.title(":material/dataset: Datensatz")
    st.divider()
    st.caption(
        "Die Seite trennt Rohdaten, Datenprüfung und die für die Modellierung genutzte Sicht. "
        "Die Ausreißeranalyse erklärt den aktuell im Projekt verwendeten IQR-Schritt "
        "für das Modeling-Dataset."
    )

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric("Rohdaten", f"{len(overview.raw_df):,}")
    with kpi_cols[1]:
        st.metric("Features", overview.raw_df.shape[1])
    with kpi_cols[2]:
        st.metric("NaN", f"{overview.total_missing:,}")
    with kpi_cols[3]:
        st.metric("Duplikate", f"{overview.duplicate_count:,}")
    with kpi_cols[4]:
        st.metric("Modeling-Dataset", f"{len(overview.modeling_df):,}")

    overview_tab, quality_tab, modeling_tab = st.tabs(
        ["Overview", "Datenprüfung", "Modeling View"]
    )

    with overview_tab:
        st.markdown("**Rohdaten auf einen Blick**")
        top_cols = st.columns([1.2, 1.2])
        with top_cols[0]:
            st.plotly_chart(build_stage_summary_chart(stage_summary), use_container_width=True)
        with top_cols[1]:
            st.dataframe(
                stage_summary.style.format(
                    {
                        "Zeilen": "{:,.0f}",
                        "Entfernte Zeilen": "{:,.0f}",
                        "Änderung %": "{:.1f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        distribution_cols = st.columns(2)
        for index, column in enumerate(["price", "mileage", "hp", "year"]):
            with distribution_cols[index % 2]:
                st.markdown(f"**Verteilung {column}**")
                st.plotly_chart(
                    build_distribution_chart(overview.raw_df.dropna(subset=[column]), column),
                    use_container_width=True,
                )

        category_cols = st.columns(2)
        category_specs = [
            ("make", 12, "Top Marken"),
            ("fuel", None, "Kraftstoff"),
            ("gear", None, "Getriebe"),
            ("offerType", None, "Angebotstyp"),
        ]
        for index, (column, top_n, title) in enumerate(category_specs):
            with category_cols[index % 2]:
                st.markdown(f"**{title}**")
                st.plotly_chart(
                    build_category_count_chart(overview.raw_df, column, top_n=top_n),
                    use_container_width=True,
                )

        with st.expander("Rohdaten-Sample", expanded=False):
            st.dataframe(overview.raw_df.head(250), use_container_width=True)

    with quality_tab:
        st.markdown("**Datenqualität und aktuelle Ausreißeranalyse**")
        st.info(
            "Aktuell werden für das Modeling-Dataset IQR-Regeln auf `price`, `mileage` "
            "und `hp` angewendet. Die Ansicht macht diesen Schritt sichtbar, statt ihn "
            "nur implizit im Preprocessing zu verstecken."
        )

        quality_cols = st.columns([1.2, 1.8])
        with quality_cols[0]:
            st.plotly_chart(build_null_counts_chart(overview.null_counts), use_container_width=True)
        with quality_cols[1]:
            st.dataframe(
                missing_summary.style.format({"NaN": "{:,.0f}", "NaN Anteil %": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
            )

        st.divider()
        factor_cols = st.columns([1, 1, 1, 2])
        with factor_cols[0]:
            factor = st.number_input("IQR-Faktor", 0.9, 2.1, 1.5, 0.1)
        cleaned_df = overview.clean_df.copy()
        analysis = analyze_outliers(cleaned_df, factor=factor)
        removed_outliers = len(analysis.merged_df) - len(analysis.filtered_df)
        with factor_cols[1]:
            st.metric("Zeilen vor Filter", f"{len(analysis.merged_df):,}")
        with factor_cols[2]:
            st.metric("Explorativ entfernte Zeilen", f"{removed_outliers:,}")
        with factor_cols[3]:
            st.metric(
                "Entfernungsquote",
                f"{removed_outliers / len(analysis.merged_df) * 100:.1f}%",
            )

        outlier_cols = st.columns([1.4, 1])
        with outlier_cols[0]:
            st.plotly_chart(
                build_outlier_chart(analysis.outlier_counts, analysis.bounds_df),
                use_container_width=True,
            )
        with outlier_cols[1]:
            st.markdown("**Numerische Kennzahlen nach Stufe**")
            comparison_summary = build_numeric_summary(
                overview.modeling_df,
                ["price", "mileage", "hp", "year"],
            )
            st.dataframe(
                comparison_summary[["Feature", "mean", "50%", "max"]].style.format(
                    {"mean": "{:.0f}", "50%": "{:.0f}", "max": "{:.0f}"}
                ),
                hide_index=True,
                use_container_width=True,
            )

    with modeling_tab:
        st.markdown("**Sicht auf den Datensatz, der aktuell in die Modellierung geht**")
        info_cols = st.columns([1.1, 1.1, 1.3])
        with info_cols[0]:
            st.metric("Rohdaten", f"{len(overview.raw_df):,}")
            st.metric("Bereinigt", f"{len(overview.clean_df):,}")
        with info_cols[1]:
            st.metric("Modeling-Dataset", f"{len(overview.modeling_df):,}")
            st.metric(
                "Behaltene Quote",
                f"{len(overview.modeling_df) / len(overview.raw_df) * 100:.1f}%",
            )
        with info_cols[2]:
            st.markdown("**Aktuelle Modellierungsannahmen**")
            st.markdown(
                "- Missing Values und Duplikate werden entfernt\n"
                "- numerische IQR-Regeln werden aktuell auf "
                "`price`, `mileage` und `hp` angewendet\n"
                "- kategoriale Spalten bleiben für die Modellierung erhalten"
            )

        detail_cols = st.columns([1.2, 1.8])
        with detail_cols[0]:
            st.markdown("**Numerische Summary**")
            st.dataframe(
                numeric_summary.style.format(
                    {
                        "count": "{:.0f}",
                        "mean": "{:.0f}",
                        "std": "{:.0f}",
                        "min": "{:.0f}",
                        "25%": "{:.0f}",
                        "50%": "{:.0f}",
                        "75%": "{:.0f}",
                        "max": "{:.0f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        with detail_cols[1]:
            st.markdown("**Modeling-Dataset Sample**")
            st.dataframe(overview.modeling_df.head(250), use_container_width=True)
