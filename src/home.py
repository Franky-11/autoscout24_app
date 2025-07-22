import streamlit as st


st.set_page_config(layout="wide")


welcome = st.Page("welcome.py", title="Welcome", icon=":material/home:", default=True)

dataset=st.Page("dataset.py", title="Datensatz", icon=":material/dataset:")


dashboard = st.Page("dashboard.py", title="Dashboard", icon=":material/dashboard:")



ml = st.Page("ml_modell.py", title="Machine Learning", icon=":material/model_training:")


pg = st.navigation([welcome,dataset,dashboard,ml])


pg.run()

st.caption("🧾 Version: v1.2.0 | Entwickler: Frank Schulnies")

