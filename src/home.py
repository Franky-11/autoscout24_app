import streamlit as st


st.set_page_config(layout="wide")


dashboard = st.Page("dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True)

dataset=st.Page("dataset.py", title="Datensatz", icon=":material/dataset:")

ml = st.Page("ml_modell.py", title="Machine Learning", icon=":material/model_training:")


pg = st.navigation([dataset,dashboard,ml])

pg.run()


