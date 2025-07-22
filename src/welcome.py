import streamlit as st

from functions import image_path


col1, col2 = st.columns([2, 1])  # Breitenverhältnis

with col1:
    st.title("")
    st.title("AutoScout24 – Data Science Projekt")

with col2:
    st.image(image_path(), use_container_width=True)



st.divider()

st.header("Über das Projekt & die Daten")
col_info,col_space=st.columns([2,1])
with col_info:
    st.info("""
    **Interaktive Web-App** für Datenanalyse & Machine Learning  
    Datengrundlage: [**Germany Cars Dataset Kaggle**](https://www.kaggle.com/datasets/ander289386/cars-germany)  
    - **Datensatz & Dashboard:**      Überblick über die Daten | interaktive Visualisierungen zur Dateninspektion.
    - **ML-Modul:**                   Auswahl verschiedener Machine Learning Modelle | Training & Modellperformance | Pipeline-Konfigurationen speichern und vergleichen
    - **Vorhersagemodul:**            Erstellen individueller Preisprognosen für spezifische Fahrzeugkonfigurationen
    - **Downloadbereich:**            Download von gespeicherten ML-Pipelines & Preisvorhersagen
    
    """)


st.divider()

# --- Sektion: Technologien & Libraries ---
st.header("Technologien & Libraries")

# Anzeige der Technologien in einer ansprechenderen Form, z.B. als Spalten oder mit Icons
techs_list = [
    ":snake: Python",
    ":card_index_dividers: Pandas",
    ":heavy_plus_sign: NumPy",
    ":gear: Scikit-Learn",
    ":zap: LightGBM",
    ":cat: CatBoost",
    ":dragon: XGBoost",
    ":bar_chart: Plotly",
    ":computer: Streamlit",
    ":straight_ruler: Scipy"]


tech_cols = st.columns(3)
for i, tech in enumerate(techs_list):
    with tech_cols[i % 3]:
        st.markdown(f"- {tech}")

st.divider()



