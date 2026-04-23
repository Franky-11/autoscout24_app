import pandas as pd
import streamlit as st

from autoscout24.modeling.config import TrainingConfig


def render_selected_feature_lists(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> None:
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Ausegewählte kategorische Features**")
        for feature in categorical_columns:
            st.markdown(f"* **{feature}**")
    with cols[1]:
        st.markdown("**Ausegewählte numerische Features**")
        for feature in numeric_columns:
            st.markdown(f"* **{feature}**")


def render_training_summary(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    selected_feature_count: int,
) -> None:
    cols = st.columns([1, 1])
    with cols[0]:
        st.metric("Anzahl Trainingsdaten", value=x_train.shape[0])
        st.metric("Anzahl Validierungsdaten", value=x_test.shape[0])
    with cols[1]:
        text = (
            "Anzahl Features"
            if selected_feature_count == x_train.shape[1]
            else "Anzahl Features mit One-Hot-Encoding"
        )
        st.metric(text, value=x_train.shape[1])


def render_parameter_summary(training_config: TrainingConfig) -> None:
    pca_text = (
        f"Aktiviert mit {training_config.n_components} Komponenten"
        if training_config.pca_enabled
        else "ohne PCA"
    )
    st.markdown("**Ausegewählte Parameter**")
    st.markdown(f"* **Modell**: {training_config.model_label}")
    st.markdown(f"* **Skalierer**: {training_config.scaler_key}")
    st.markdown(f"* **Target**: {training_config.target_transform_label}")
    st.markdown(f"* **PCA**: {pca_text}")
    st.markdown(f"* **CV Folds**: {training_config.cv_folds}")


def render_prediction_inputs(
    df: pd.DataFrame,
    required_widgets: list[str],
    selected_features: list[str],
) -> dict[str, object]:
    values: dict[str, object] = {
        "hp": None,
        "year": None,
        "mileage": None,
        "make": None,
        "fuel": None,
        "gear": None,
        "model": None,
        "offerType": None,
    }

    cols = st.columns(3)
    with cols[0]:
        if "hp" in required_widgets:
            values["hp"] = st.slider(
                "PS",
                min_value=int(df["hp"].min()),
                max_value=int(df["hp"].max()),
                value=135,
                step=5,
            )
        if "year" in required_widgets:
            values["year"] = st.number_input("Erstzulassung", 2011, 2021, 2015, step=1)
        if "mileage" in required_widgets:
            values["mileage"] = st.slider(
                "Kilometerstand",
                min_value=0,
                max_value=int(df["mileage"].max()),
                value=15000,
                step=1000,
            )

    with cols[1]:
        if "make" in required_widgets:
            values["make"] = st.selectbox("Marke", sorted(df["make"].unique()), index=5)
        if "model" in required_widgets:
            if "make" in selected_features and values["make"] is not None:
                model_options = sorted(df[df["make"] == values["make"]]["model"].unique())
            else:
                model_options = sorted(df["model"].unique())
            values["model"] = st.selectbox("Modell", model_options, index=0)
        if "offerType" in required_widgets:
            values["offerType"] = st.selectbox(
                "Angebotstyp",
                sorted(df["offerType"].unique()),
                index=0,
            )

    with cols[2]:
        if "fuel" in required_widgets:
            values["fuel"] = st.segmented_control(
                "Kraftstoff",
                options=sorted(df["fuel"].unique()),
                default="Gasoline",
            )
        if "gear" in required_widgets:
            values["gear"] = st.segmented_control(
                "Getriebe",
                options=sorted(df["gear"].unique()),
                default="Manual",
            )

    return values
