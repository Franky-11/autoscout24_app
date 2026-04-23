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
from autoscout24.data.io import load_modeling_dataset
from autoscout24.features.engineering import (
    build_vehicle_input_frame,
    get_engineered_feature_names,
    required_prediction_inputs,
)
from autoscout24.modeling.config import (
    DEFAULT_FEATURE_SELECTION,
    FeatureSelectionConfig,
    TrainingConfig,
)
from autoscout24.modeling.evaluation import (
    SEGMENT_LABELS,
    build_pca_explained_variance_chart,
    build_prediction_scatter,
    build_qq_plot,
    build_segment_error_chart,
    calculate_feature_importance,
)
from autoscout24.modeling.registry import persist_run
from autoscout24.modeling.service import (
    MODEL_LABELS,
    calculate_pca_explained_variance,
    predict_with_pipeline,
    prepare_training_data,
    train_model_run,
)
from autoscout24.modeling.tuning import compare_model_candidates


@st.cache_data
def _load_modeling_base_df():
    return load_modeling_dataset()


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
    st.session_state.evaluation_report = run_result.evaluation_report
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
        "train_time": st.session_state.train_time,
        "selected_features": feature_config.all_features,
        "feature_columns": prepared_data.feature_frame.columns.tolist(),
        "categorical_columns": prepared_data.categorical_columns,
        "numeric_columns": prepared_data.numeric_columns,
        "evaluation_report": st.session_state.evaluation_report.to_dict(),
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


def _render_model_comparison(
    prepared_data,
    training_config: TrainingConfig,
) -> None:
    st.divider()
    st.subheader("🧪 Modellvergleich")
    comparison_cols = st.columns([2, 1])
    with comparison_cols[0]:
        comparison_models = st.multiselect(
            "Kandidaten",
            options=list(MODEL_LABELS),
            default=list(dict.fromkeys([training_config.model_key, "cat", "lgb"])),
            format_func=lambda key: MODEL_LABELS[key],
        )
    with comparison_cols[1]:
        max_candidates_per_model = st.slider("Configs pro Modell", 1, 4, 2)

    if st.button("Modellvergleich starten"):
        with st.spinner("Vergleiche Kandidaten per Cross-Validation..."):
            st.session_state.comparison_results = compare_model_candidates(
                prepared_data,
                model_keys=comparison_models,
                target_transform=training_config.target_transform,
                cv_folds=training_config.cv_folds,
                max_candidates_per_model=max_candidates_per_model,
            )

    if st.session_state.comparison_results.empty:
        return

    best_candidate = st.session_state.comparison_results.iloc[0]
    comparison_metrics = st.columns(3)
    with comparison_metrics[0]:
        st.metric("Bestes Modell", best_candidate["Model"])
    with comparison_metrics[1]:
        st.metric("Beste CV RMSE", f"{best_candidate['CV RMSE']:.0f}")
    with comparison_metrics[2]:
        st.metric("Beste CV MAE", f"{best_candidate['CV MAE']:.0f}")

    st.dataframe(
        st.session_state.comparison_results.style.format(
            {
                "CV RMSE": "{:.0f}",
                "CV RMSE Std": "{:.0f}",
                "CV MAE": "{:.0f}",
                "CV MAE Std": "{:.0f}",
                "CV R2": "{:.3f}",
                "CV R2 Std": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_residual_analysis() -> None:
    evaluation_report = st.session_state.evaluation_report
    diagnostics = evaluation_report.residual_diagnostics

    with st.popover("Residuen Analyse", use_container_width=True):
        tab_qq, tab_price, tab_make, tab_offer, tab_age = st.tabs(
            [
                "QQ-Plot",
                "Preisband",
                "Marke",
                "Angebotstyp",
                "Fahrzeugalter",
            ]
        )
        with tab_qq:
            skew_cols = st.columns(2)
            with skew_cols[0]:
                st.metric(
                    label="Schiefe (Skewness)",
                    value=f"{diagnostics.skew:.2f}",
                    delta=f"{diagnostics.delta_skew:.2f} zur Normalform",
                    delta_color="inverse" if diagnostics.delta_skew < 0.5 else "off",
                )
            with skew_cols[1]:
                st.metric(
                    label="Spitzigkeit (Kurtosis)",
                    value=f"{diagnostics.kurtosis:.2f}",
                    delta=f"{diagnostics.delta_kurtosis:.2f} zur Normalform",
                    delta_color="inverse" if diagnostics.delta_kurtosis < 0.5 else "off",
                )
            st.plotly_chart(
                build_qq_plot(evaluation_report.y_true, evaluation_report.y_pred),
                use_container_width=True,
            )
        for tab, report_key, top_n in [
            (tab_price, "price_band", None),
            (tab_make, "make", 15),
            (tab_offer, "offerType", None),
            (tab_age, "car_age_band", None),
        ]:
            with tab:
                segment_df = evaluation_report.segment_reports[report_key]
                chart_frame = segment_df.head(top_n) if top_n else segment_df
                st.plotly_chart(
                    build_segment_error_chart(
                        chart_frame,
                        title=f"Fehler nach {SEGMENT_LABELS[report_key]}",
                    ),
                    use_container_width=True,
                )
                st.dataframe(
                    segment_df.style.format(
                        {
                            "MAE": "{:.0f}",
                            "Median AE": "{:.0f}",
                            "Bias": "{:.0f}",
                            "RMSE": "{:.0f}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )


def _render_training_results(
    prepared_data,
    feature_config: FeatureSelectionConfig,
    training_config: TrainingConfig,
) -> None:
    st.divider()
    st.subheader("📈 Modell trainieren und evaluieren")
    render_training_summary(
        prepared_data.x_train,
        prepared_data.x_test,
        len(feature_config.all_features),
    )

    placeholder = st.empty()
    scenario_exists = config_exists(
        st.session_state.scenario_log,
        feature_config,
        training_config,
    )
    if scenario_exists:
        placeholder.info("Modellsetup ist bereits im Session-Log vorhanden.")
    if not st.session_state.model_training:
        placeholder.info("Modell noch nicht trainiert.")

    if st.button("Modell trainieren"):
        with st.spinner("Training läuft... Dies kann einen Moment dauern."):
            _run_training(prepared_data, feature_config, training_config)
        placeholder.success("Training abgeschlossen!")

    if not st.session_state.model_training:
        return

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

            _render_residual_analysis()

            st.plotly_chart(
                build_prediction_scatter(
                    st.session_state.evaluation_report.y_true,
                    st.session_state.evaluation_report.y_pred,
                ),
                use_container_width=True,
            )

            baseline_df = pd.DataFrame(
                [
                    {
                        "Baseline": baseline.name,
                        "RMSE": baseline.rmse,
                        "R2": baseline.r2,
                        "MAE": baseline.mae,
                    }
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
            importance_df = calculate_feature_importance(
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


def _render_prediction_section(
    prepared_data,
    feature_config: FeatureSelectionConfig,
) -> None:
    if not st.session_state.model_training:
        return

    st.divider()
    st.subheader("💸 Preis vorhersagen")
    st.markdown(
        f"""
        **Pipeline-Konfiguration für Vorhersage**
        - Skalierer: **{st.session_state.scaler}**
        - Target: **{st.session_state.target_transform}**
        - Modell: **{MODEL_LABELS[st.session_state.model]}**
        """
    )

    required_widgets = required_prediction_inputs(st.session_state.features)
    with st.container(border=True, height=620):
        prediction_inputs = render_prediction_inputs(
            prepared_data.engineered_df,
            required_widgets,
            feature_config.all_features,
        )
        new_car_df = build_vehicle_input_frame(
            vehicle_inputs={
                "hp": prediction_inputs["hp"],
                "year": prediction_inputs["year"],
                "mileage": prediction_inputs["mileage"],
                "make": prediction_inputs["make"],
                "fuel": prediction_inputs["fuel"],
                "gear": prediction_inputs["gear"],
                "model": prediction_inputs["model"],
                "offerType": prediction_inputs["offerType"],
            },
            selected_features=st.session_state.features,
            training_columns=st.session_state.X_train.columns.tolist(),
            categorical_columns=prepared_data.categorical_columns,
            model_selection=st.session_state.model_selection,
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


def render_page() -> None:
    initialize_modeling_state()

    st.title(":material/model_training: Machine Learning zur Preisvorhersage")
    st.divider()

    base_df = _load_modeling_base_df()
    feature_options = [feature for feature in base_df.columns.tolist() if feature != "price"]
    engineered_feature_options = get_engineered_feature_names()

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

    explained_variance = calculate_pca_explained_variance(
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
            st.plotly_chart(
                build_pca_explained_variance_chart(explained_variance),
                use_container_width=True,
            )
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
        st.session_state.comparison_results = pd.DataFrame()

    _render_model_comparison(prepared_data, training_config)
    _render_training_results(prepared_data, feature_config, training_config)
    _render_prediction_section(prepared_data, feature_config)

    st.divider()
    render_download_section(
        st.session_state.scenario_log,
        st.session_state.car_price,
        st.session_state.pipeline_store,
    )
