import streamlit as st

from autoscout24.exploration.charts import build_null_counts_chart, build_outlier_chart
from autoscout24.exploration.service import analyze_outliers, load_dataset_overview


@st.cache_data
def _load_dataset_overview():
    return load_dataset_overview()


def render_page() -> None:
    overview = _load_dataset_overview()

    st.title(":material/dataset: Datensatz")
    st.divider()
    st.markdown(
        f"**Germany Cars Dataset from kaggle** "
        f"&nbsp;&nbsp;|&nbsp;&nbsp;{overview.raw_df.shape[0]} Autos "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; {overview.raw_df.shape[1]} Features"
    )
    st.dataframe(overview.raw_df)

    st.divider()
    st.markdown(
        f"**NaN Werte**&nbsp;&nbsp;|&nbsp;&nbsp;{overview.total_missing}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;**Duplikate**&nbsp;&nbsp;|&nbsp;&nbsp;{overview.duplicate_count}"
    )
    st.plotly_chart(build_null_counts_chart(overview.null_counts), use_container_width=True)

    st.divider()
    st.markdown("**Entfernen von numerischen Ausreißern mittels IQR**")
    cols = st.columns([1, 0.5, 2])
    with cols[0]:
        st.markdown(
            """
            * Berechnen von Q1 und Q3 sowie IQR
            * Festlegen oberes Limit: Q3 + 1.5 × IQR
            * Filtern der Daten nach oberem Limit
            """
        )
        st.write("")
        st.write("")
        factor = st.number_input("Faktor", 0.9, 2.1, 1.5, 0.1)
        cleaned_df = overview.raw_df.dropna().drop_duplicates().copy()
        analysis = analyze_outliers(cleaned_df, factor=factor)
        st.markdown(f"{analysis.outlier_counts['outliers'].sum()} Ausreißer")

    with cols[2]:
        st.plotly_chart(
            build_outlier_chart(analysis.outlier_counts, analysis.bounds_df),
            use_container_width=True,
        )
