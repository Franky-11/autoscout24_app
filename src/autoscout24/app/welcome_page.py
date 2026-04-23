import streamlit as st

from autoscout24.data.io import get_image_path

TECH_STACK = [
    ":snake: Python",
    ":card_index_dividers: Pandas",
    ":heavy_plus_sign: NumPy",
    ":gear: Scikit-Learn",
    ":zap: LightGBM",
    ":cat: CatBoost",
    ":dragon: XGBoost",
    ":bar_chart: Plotly",
    ":computer: Streamlit",
    ":straight_ruler: Scipy",
]


def render_page() -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("AutoScout24 – Preisanalyse und ML-App")
    with col2:
        st.image(get_image_path(), use_container_width=True)

    st.divider()
    st.header("Was die App zeigt")
    intro_cols = st.columns([1.3, 1.2])
    with intro_cols[0]:
        st.info(
            """
            Die Anwendung kombiniert Datensatzsicht, interaktive Exploration und
            ein reproduzierbares Modeling-Setup für Fahrzeugpreise.

            Datengrundlage:
            [Germany Cars Dataset on Kaggle](https://www.kaggle.com/datasets/ander289386/cars-germany)
            """
        )
    with intro_cols[1]:
        st.markdown(
            """
            **Seiten**
            - `Datensatz`: Rohdaten und Modeling-Sicht
            - `Dashboard`: interaktive Exploration nach Marke, Kraftstoff und Getriebe
            - `Machine Learning`: Setup, Screening, finales Training und Vorhersage
            """
        )

    st.divider()
    st.header("Modeling-Workflow")
    workflow_cols = st.columns(3)
    workflow_steps = [
        (
            "1. Fester Input",
            "Vor dem Training werden NaN-Zeilen, Duplikate und unplausible Werte entfernt. "
            "Es gibt keinen interaktiven Outlier-Filter auf `price`.",
        ),
        (
            "2. Kandidaten Vergleichen",
            "Im Screening werden mehrere Modellkandidaten per Cross-Validation verglichen. "
            "Ein Kandidat kann explizit ins finale Training übernommen werden.",
        ),
        (
            "3. Run Persistieren",
            "Final trainierte Runs lassen sich speichern, später wieder laden und für "
            "Vorhersagen oder Downloads weiterverwenden.",
        ),
    ]
    for column, (title, body) in zip(workflow_cols, workflow_steps, strict=False):
        with column:
            st.markdown(f"**{title}**")
            st.write(body)

    st.divider()
    st.header("Technologien")
    tech_cols = st.columns(3)
    for index, tech in enumerate(TECH_STACK):
        with tech_cols[index % 3]:
            st.markdown(f"- {tech}")
    st.divider()
