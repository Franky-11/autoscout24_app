import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.utils import estimator_html_repr

from autoscout24.app.modeling_exports import render_download_section
from autoscout24.app.modeling_sections import (
    render_parameter_summary,
    render_prediction_inputs,
    render_selected_feature_lists,
    render_training_summary,
)
from autoscout24.app.modeling_state import (
    build_config_signature,
    config_exists,
    initialize_modeling_state,
)
from autoscout24.modeling.config import DEFAULT_FEATURE_SELECTION, FeatureSelectionConfig, TrainingConfig
from autoscout24.modeling.registry import persist_run
from autoscout24.modeling.service import (
    MODEL_LABELS,
    predict_with_pipeline,
    prepare_training_data,
    train_model_run,
)
from ml_functions import (
    car_data,
    check_residuals,
    error_price_segments,
    feature_engineering,
    feature_importance,
    input_features,
    plot_explained_var,
    plot_pred_vs_true,
    pca_explained_variance,
    qq_plot,
    read_preprocess_df,
)


def _run_training(
    prepared_data,
    feature_config: FeatureSelectionConfig,
    training_config: TrainingConfig,
) -> None:
    run_result = train_model_run(
        prepared_data=prepared_data,
        model_selection=training_config.model_key,
        scaler_selection=training_config.scaler_key,
        pca_check=training_config.pca_enabled,
        n_components=training_config.n_components,
        target_transform=training_config.target_transform,
        cv_folds=training_config.cv_folds,
    )

    st.session_state.model_training = True
    st.session_state.show_scenarios_popover = False
    st.session_state.train_time = run_result.train_time
    st.session_state.X_train = prepared_data.x_train
    st.session_state.y_test = prepared_data.y_test
    st.session_state.y_pred = run_result.y_pred
    st.session_state.rmse = run_result.rmse
    st.session_state.r2 = run_result.r2
    st.session_state.mae = run_result.mae
    st.session_state.cv_summary = run_result.cv_summary
    st.session_state.baselines = run_result.baselines
    st.session_state.pipe = run_result.pipeline
    st.session_state.model_obj = run_result.model
    st.session_state.model_selection = training_config.model_key
    st.session_state.categorical_dummy_cols = prepared_data.categorical_dummy_columns
    st.session_state.features = feature_config.all_features
    st.session_state.model = training_config.model_key
    st.session_state.scaler = training_config.scaler_key
    st.session_state.pca_check = training_config.pca_enabled
    st.session_state.n_components = training_config.n_components
    st.session_state.test_size = training_config.test_size
    st.session_state.target_transform = training_config.target_transform
    st.session_state.cv_folds = training_config.cv_folds
    st.session_state.last_config_signature = build_config_signature(
        feature_config,
        training_config,
    )


def _save_current_pipeline(
    prepared_data,
    feature_config: FeatureSelectionConfig,
    training_config: TrainingConfig,
) -> None:
    new_scenario_id = len(st.session_state.scenario_log) + 1
    model_params = st.session_state.model_obj.get_params()
    relevant_params = {
        key: value
        for key, value in model_params.items()
        if not key.startswith(("regressor__", "alpha", "random_state", "n_jobs"))
    }
    best_baseline_rmse = min(baseline.rmse for baseline in st.session_state.baselines)
    metadata = {
        "feature_config": feature_config,
        "training_config": training_config,
        "holdout_metrics": {
            "rmse": st.session_state.rmse,
            "r2": st.session_state.r2,
            "mae": st.session_state.mae,
        },
        "cross_validation": st.session_state.cv_summary,
        "baselines": st.session_state.baselines,
        "model_parameters": relevant_params,
        "feature_columns": prepared_data.feature_frame.columns.tolist(),
    }
    persisted_run = persist_run(
        pipeline=st.session_state.pipe,
        metadata=metadata,
        run_id=f"run-{new_scenario_id:03d}",
    )

    new_row = {
        "Pipe ID": new_scenario_id,
        "Run ID": persisted_run.run_id,
        "Features": feature_config.all_features,
        "Anzahl Features": len(prepared_data.feature_frame.columns),
        "Model": training_config.model_label,
        "Scaler": training_config.scaler_key,
        "Target": training_config.target_transform,
        "PCA Active": training_config.pca_enabled,
        "PCA Components": training_config.effective_pca_components,
        "CV Folds": training_config.cv_folds,
        "Test Size": training_config.test_size,
        "Model Parameters": str(relevant_params),
        "Train Time (s)": st.session_state.train_time,
        "R2 Score": st.session_state.r2,
        "RMSE": st.session_state.rmse,
        "MAE": st.session_state.mae,
        "CV RMSE": st.session_state.cv_summary.mean_rmse,
        "CV MAE": st.session_state.cv_summary.mean_mae,
        "Best Baseline RMSE": best_baseline_rmse,
        "Artifact Path": str(persisted_run.run_dir),
    }
    st.session_state.scenario_log = pd.concat(
        [st.session_state.scenario_log, pd.DataFrame([new_row])],
        ignore_index=True,
    )
    st.session_state.pipeline_store[new_scenario_id] = st.session_state.pipe
    st.session_state.show_scenarios_popover = True
    st.success("Pipeline gespeichert und Artefakte persistiert.")


initialize_modeling_state()

st.title(":material/model_training: Machine Learning zur Preisvorhersage")
st.divider()

base_df = read_preprocess_df()
engineered_df, engineered_feature_options = feature_engineering(base_df)
feature_options = [feature for feature in base_df.columns.tolist() if feature != "price"]

st.subheader("🛠️ Feature Auswahl")
feature_cols = st.columns([1.5, 1.5, 2])
with feature_cols[0]:
    base_features = st.multiselect(
        "Features",
        options=feature_options,
        default=DEFAULT_FEATURE_SELECTION.base_features,
    )
with feature_cols[1]:
    engineered_features = st.multiselect(
        "Engineered Features",
        options=engineered_feature_options,
        default=DEFAULT_FEATURE_SELECTION.engineered_features,
    )

feature_config = FeatureSelectionConfig(
    base_features=base_features,
    engineered_features=engineered_features,
)
if not feature_config.all_features:
    st.warning("Bitte mindestens ein Feature auswählen.")
    st.stop()

st.divider()
st.subheader("⚙️ Modell Auswahl und Preprocessing")

config_cols = st.columns([1.4, 1.4, 1.2, 1.2, 1.4])
with config_cols[0]:
    model_selection = st.selectbox(
        "Wähle ein Regressionsmodell:",
        ("lr", "rf", "xgb", "cat", "lgb"),
        format_func=lambda key: MODEL_LABELS[key],
    )
with config_cols[1]:
    if model_selection == "lr":
        scaler_selection = st.segmented_control(
            "Skalierer",
            options=["None", "Standard", "MinMax"],
            default="Standard",
        )
    else:
        scaler_selection = "None"
        st.badge("Skalierung nicht erforderlich", color="green")
with config_cols[2]:
    target_transform = st.selectbox(
        "Target",
        options=["raw", "log1p"],
        format_func=lambda key: {"raw": "Preis direkt", "log1p": "log1p(Preis)"}[key],
    )
    test_size = st.slider("Anteil Validierungsset", 10, 50, 20, 5) / 100
with config_cols[3]:
    cv_folds = st.slider("CV Folds", 3, 5, 3, 1)
    pca_check = model_selection == "lr" and st.checkbox("PCA aktivieren?", value=False)
with config_cols[4]:
    n_components = 10

training_config = TrainingConfig(
    model_key=model_selection,
    scaler_key=scaler_selection,
    pca_enabled=pca_check,
    n_components=n_components,
    test_size=test_size,
    target_transform=target_transform,
    cv_folds=cv_folds,
)
prepared_data = prepare_training_data(
    base_features=feature_config.base_features,
    engineered_features=feature_config.engineered_features,
    model_selection=training_config.model_key,
    test_size=training_config.test_size,
)

render_selected_feature_lists(
    prepared_data.categorical_columns,
    prepared_data.numeric_columns,
)

explained_variance = pca_explained_variance(
    prepared_data.x_train,
    training_config.model_key,
    training_config.scaler_key,
    prepared_data.categorical_dummy_columns,
    prepared_data.numeric_columns,
    prepared_data.categorical_columns,
)
if training_config.pca_enabled and explained_variance is not None:
    with config_cols[4]:
        n_opt = int((explained_variance >= 0.95).argmax()) + 1
        max_components = len(prepared_data.numeric_columns) + len(
            prepared_data.categorical_dummy_columns
        ) - 1
        n_components = st.slider(
            "Anzahl der PCA-Komponenten",
            min_value=1,
            max_value=max_components,
            value=min(n_opt, max_components),
        )
        st.write(f"▶️ Optimale PCA-Komponenten für 95% erklärte Varianz: {n_opt}")
        st.plotly_chart(plot_explained_var(explained_variance), use_container_width=True)
    training_config = TrainingConfig(
        model_key=model_selection,
        scaler_key=scaler_selection,
        pca_enabled=pca_check,
        n_components=n_components,
        test_size=test_size,
        target_transform=target_transform,
        cv_folds=cv_folds,
    )

render_parameter_summary(training_config)

current_signature = build_config_signature(feature_config, training_config)
scenario_exists = config_exists(
    st.session_state.scenario_log,
    feature_config,
    training_config,
)
parameters_changed = st.session_state.last_config_signature != current_signature
if parameters_changed:
    st.session_state.model_training = False
    st.session_state.show_scenarios_popover = scenario_exists

st.divider()
st.subheader("📈 Modell trainieren und evaluieren")
render_training_summary(
    prepared_data.x_train,
    prepared_data.x_test,
    len(feature_config.all_features),
)

placeholder = st.empty()
if scenario_exists:
    placeholder.info("Modellsetup ist bereits im Session-Log vorhanden.")
if not st.session_state.model_training:
    placeholder.info("Modell noch nicht trainiert.")

if st.button("Modell trainieren"):
    with st.spinner("Training läuft... Dies kann einen Moment dauern."):
        _run_training(prepared_data, feature_config, training_config)
    placeholder.success("Training abgeschlossen!")

if st.session_state.model_training:
    with st.expander("Pipeline-Struktur", expanded=False):
        components.html(
            estimator_html_repr(st.session_state.pipe),
            height=300,
            scrolling=True,
        )

    with st.container(border=True, height=1080):
        cols = st.columns([3, 0.5, 1.2])
        with cols[0]:
            st.markdown(
                f"**Modell Performance {MODEL_LABELS[st.session_state.model_selection]} "
                "auf dem Validierungsset**"
            )
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("RMSE", value=f"{st.session_state.rmse:.0f}")
                st.metric("CV RMSE", value=f"{st.session_state.cv_summary.mean_rmse:.0f}")
            with metric_cols[1]:
                st.metric("R2", value=f"{st.session_state.r2:.3f}")
                st.metric("CV R2", value=f"{st.session_state.cv_summary.mean_r2:.3f}")
            with metric_cols[2]:
                st.metric("MAE", value=f"{st.session_state.mae:.0f}")
                st.metric("CV MAE", value=f"{st.session_state.cv_summary.mean_mae:.0f}")

            qq = qq_plot(st.session_state.y_test, st.session_state.y_pred)
            res_skew, res_kurt, delta_skew, delta_kurt = check_residuals(
                st.session_state.y_test,
                st.session_state.y_pred,
            )

            with st.popover("Residuen Analyse", use_container_width=True):
                tab_qq, tab_error = st.tabs(["QQ-Plot", "Fehler nach Preissegmenten"])
                with tab_qq:
                    skew_cols = st.columns(2)
                    with skew_cols[0]:
                        st.metric(
                            label="Schiefe (Skewness)",
                            value=f"{res_skew:.2f}",
                            delta=f"{delta_skew:.2f} zur Normalform",
                            delta_color="inverse" if delta_skew < 0.5 else "off",
                        )
                    with skew_cols[1]:
                        st.metric(
                            label="Spitzigkeit (Kurtosis)",
                            value=f"{res_kurt:.2f}",
                            delta=f"{delta_kurt:.2f} zur Normalform",
                            delta_color="inverse" if delta_kurt < 0.5 else "off",
                        )
                    st.plotly_chart(qq, use_container_width=True)
                with tab_error:
                    st.plotly_chart(
                        error_price_segments(st.session_state.y_test, st.session_state.y_pred),
                        use_container_width=True,
                    )

            st.plotly_chart(
                plot_pred_vs_true(st.session_state.y_test, st.session_state.y_pred),
                use_container_width=True,
            )

            baseline_df = pd.DataFrame(
                [
                    {"Baseline": baseline.name, "RMSE": baseline.rmse, "R2": baseline.r2, "MAE": baseline.mae}
                    for baseline in st.session_state.baselines
                ]
            )
            st.markdown("**Holdout-Baselines**")
            st.dataframe(
                baseline_df.style.format({"RMSE": "{:.0f}", "R2": "{:.3f}", "MAE": "{:.0f}"}),
                hide_index=True,
                use_container_width=True,
            )

        with cols[2]:
            st.markdown("**Feature Importance**")
            importance_df = feature_importance(
                st.session_state.pipe,
                st.session_state.model_selection,
                st.session_state.X_train,
            )
            st.dataframe(
                importance_df.style.format({"Importance": "{:.2f}", "Relative%": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("Pipeline speichern"):
            _save_current_pipeline(prepared_data, feature_config, training_config)

        if st.session_state.show_scenarios_popover:
            with st.popover("Gespeicherte Pipelines", use_container_width=True):
                st.dataframe(
                    st.session_state.scenario_log.style.format(
                        {
                            "Test Size": "{:.2f}",
                            "Train Time (s)": "{:.2f}",
                            "R2 Score": "{:.3f}",
                            "RMSE": "{:.0f}",
                            "MAE": "{:.0f}",
                            "CV RMSE": "{:.0f}",
                            "CV MAE": "{:.0f}",
                            "Best Baseline RMSE": "{:.0f}",
                        }
                    ),
                    use_container_width=True,
                )

st.divider()

if st.session_state.model_training:
    st.subheader("💸 Preis vorhersagen")
    st.markdown(
        f"""
        **Pipeline-Konfiguration für Vorhersage**
        - Skalierer: **{st.session_state.scaler}**
        - Target: **{st.session_state.target_transform}**
        - Modell: **{MODEL_LABELS[st.session_state.model]}**
        """
    )

    required_widgets = input_features(st.session_state.features)
    with st.container(border=True, height=620):
        prediction_inputs = render_prediction_inputs(
            prepared_data.engineered_df,
            required_widgets,
            feature_config.all_features,
        )
        new_car_df = car_data(
            st.session_state.model_selection,
            st.session_state.categorical_dummy_cols,
            st.session_state.X_train,
            prepared_data.categorical_columns,
            st.session_state.features,
            hp=prediction_inputs["hp"],
            year=prediction_inputs["year"],
            mileage=prediction_inputs["mileage"],
            make=prediction_inputs["make"],
            fuel=prediction_inputs["fuel"],
            gear=prediction_inputs["gear"],
            model=prediction_inputs["model"],
            offerType=prediction_inputs["offerType"],
        )
        car_price = predict_with_pipeline(
            st.session_state.pipe,
            new_car_df,
            st.session_state.target_transform,
        )

        prediction_row = {
            "Pipe ID": (
                st.session_state.scenario_log["Pipe ID"].iloc[-1]
                if not st.session_state.scenario_log.empty
                else "N/A"
            ),
            "Run ID": (
                st.session_state.scenario_log["Run ID"].iloc[-1]
                if not st.session_state.scenario_log.empty
                else "N/A"
            ),
            "Marke": prediction_inputs["make"],
            "Modell": prediction_inputs["model"],
            "PS": prediction_inputs["hp"],
            "Erstzulassung": prediction_inputs["year"],
            "Kilometerstand": prediction_inputs["mileage"],
            "Kraftstoff": prediction_inputs["fuel"],
            "Getriebe": prediction_inputs["gear"],
            "Angebotstyp": prediction_inputs["offerType"],
            "Vorhergesagter Preis": f"{car_price[0]:.0f}",
        }

        pred_cols = st.columns([0.6, 1.2, 1.6])
        with pred_cols[0]:
            st.metric("Vorhergesagter Preis", value=f"{car_price[0]:.0f}")
        with pred_cols[1]:
            if not st.session_state.scenario_log.empty and st.button("Preisvorhersage speichern"):
                st.session_state.car_price = pd.concat(
                    [st.session_state.car_price, pd.DataFrame([prediction_row])],
                    ignore_index=True,
                )
                st.success("Preisvorhersage gespeichert!")
        with pred_cols[2]:
            with st.popover("Gespeicherte Vorhersagen", use_container_width=True):
                st.dataframe(st.session_state.car_price, use_container_width=True)

st.divider()
render_download_section(
    st.session_state.scenario_log,
    st.session_state.car_price,
    st.session_state.pipeline_store,
)
