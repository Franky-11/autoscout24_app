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
        st.title("")
        st.title("AutoScout24 – Data Science Projekt")
    with col2:
        st.image(get_image_path(), use_container_width=True)

    st.divider()
    st.header("Über das Projekt & die Daten")
    col_info, _ = st.columns([2, 1])
    with col_info:
        st.info(
            """
            **Interaktive Web-App** für Datenanalyse & Machine Learning
            Datengrundlage: [**Germany Cars Dataset Kaggle**](https://www.kaggle.com/datasets/ander289386/cars-germany)
            - **Datensatz & Dashboard:** Überblick über die Daten |
              interaktive Visualisierungen zur Dateninspektion
            - **ML-Modul:** Auswahl verschiedener Machine Learning Modelle |
              Training & Modellperformance | Pipeline-Konfigurationen speichern
              und vergleichen
            - **Vorhersagemodul:** Erstellen individueller Preisprognosen
              für spezifische Fahrzeugkonfigurationen
            - **Downloadbereich:** Download von gespeicherten ML-Pipelines & Preisvorhersagen
            """
        )

    st.divider()
    st.header("Technologien & Libraries")
    tech_cols = st.columns(3)
    for index, tech in enumerate(TECH_STACK):
        with tech_cols[index % 3]:
            st.markdown(f"- {tech}")
    st.divider()
