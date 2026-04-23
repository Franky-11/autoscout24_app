import streamlit as st

from autoscout24.app.dashboard_page import render_page as render_dashboard_page
from autoscout24.app.dataset_page import render_page as render_dataset_page
from autoscout24.app.modeling_page import render_page as render_modeling_page
from autoscout24.app.welcome_page import render_page as render_welcome_page


def run_app() -> None:
    st.set_page_config(layout="wide")

    welcome = st.Page(
        render_welcome_page,
        title="Welcome",
        icon=":material/home:",
        url_path="welcome",
        default=True,
    )
    dataset = st.Page(
        render_dataset_page,
        title="Datensatz",
        icon=":material/dataset:",
        url_path="dataset",
    )
    dashboard = st.Page(
        render_dashboard_page,
        title="Dashboard",
        icon=":material/dashboard:",
        url_path="dashboard",
    )
    ml = st.Page(
        render_modeling_page,
        title="Machine Learning",
        icon=":material/model_training:",
        url_path="machine-learning",
    )

    navigation = st.navigation([welcome, dataset, dashboard, ml])
    navigation.run()

    st.caption("🧾 Version: v1.2.3 | Entwickler: Frank Schulnies")
