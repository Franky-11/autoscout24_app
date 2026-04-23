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
    clear_screening_candidate,
    config_exists,
    consume_pending_screening_candidate,
    initialize_modeling_state,
    queue_screening_candidate,
    screening_candidate_still_matches,
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
        model_params=st.session_state.setup_model_params,
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
    st.subheader("🧪 Kandidaten-Screening")
    st.info(
        "Das Screening vergleicht mehrere Kandidaten per Cross-Validation. "
        "Dabei wird noch kein finaler Run für Vorhersage, Persistenz oder Export erzeugt."
    )
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
    selected_candidate_index = st.selectbox(
        "Kandidat für Übernahme",
        options=st.session_state.comparison_results.index.tolist(),
        format_func=lambda index: (
            f"#{index + 1} "
            f"{st.session_state.comparison_results.loc[index, 'Model']} | "
            f"RMSE {st.session_state.comparison_results.loc[index, 'CV RMSE']:.0f}"
        ),
    )
    comparison_metrics = st.columns(3)
    with comparison_metrics[0]:
        st.metric("Bestes Modell", best_candidate["Model"])
    with comparison_metrics[1]:
        st.metric("Beste CV RMSE", f"{best_candidate['CV RMSE']:.0f}")
    with comparison_metrics[2]:
        st.metric("Beste CV MAE", f"{best_candidate['CV MAE']:.0f}")

    action_cols = st.columns([1.4, 1.6, 3])
    with action_cols[0]:
        if st.button("Besten Kandidaten übernehmen"):
            queue_screening_candidate(best_candidate.to_dict())
            st.rerun()
    with action_cols[1]:
        if st.button("Ausgewählten Kandidaten übernehmen"):
            candidate = st.session_state.comparison_results.loc[selected_candidate_index]
            queue_screening_candidate(candidate.to_dict())
            st.rerun()

    st.dataframe(
        st.session_state.comparison_results.drop(columns=["Parameters Raw"]).style.format(
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


def _render_residual_analysis_tab() -> None:
    evaluation_report = st.session_state.evaluation_report
    diagnostics = evaluation_report.residual_diagnostics

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

    segment_tabs = st.tabs(["QQ-Plot", "Preisband", "Marke", "Angebotstyp", "Fahrzeugalter"])
    with segment_tabs[0]:
        st.plotly_chart(
            build_qq_plot(evaluation_report.y_true, evaluation_report.y_pred),
            use_container_width=True,
        )
    for tab, report_key, top_n in [
        (segment_tabs[1], "price_band", None),
        (segment_tabs[2], "make", 15),
        (segment_tabs[3], "offerType", None),
        (segment_tabs[4], "car_age_band", None),
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


def _render_training_tab(
    prepared_data,
    feature_config: FeatureSelectionConfig,
    training_config: TrainingConfig,
) -> None:
    st.subheader("📈 Finales Training")
    st.info(
        "Hier wird genau ein Setup final auf dem aktuellen Trainingssplit trainiert. "
        "Erst dieser Schritt erzeugt ein Modell für Diagnose, Vorhersage und Speichern."
    )
    render_training_summary(
        prepared_data.x_train,
        prepared_data.x_test,
        len(feature_config.all_features),
    )
    scenario_exists = config_exists(
        st.session_state.scenario_log,
        feature_config,
        training_config,
    )
    if scenario_exists:
        st.info("Dieses Modellsetup ist bereits im Session-Log vorhanden.")
    if not st.session_state.model_training:
        st.info("Nach Konfigurationsänderungen ist noch kein Training für dieses Setup gelaufen.")

    if st.session_state.setup_candidate_label:
        st.caption(f"Aktiver Screening-Kandidat: {st.session_state.setup_candidate_label}")
        st.code(str(st.session_state.setup_model_params), language="python")

    if st.button("Modell trainieren", type="primary"):
        with st.spinner("Training läuft... Dies kann einen Moment dauern."):
            _run_training(prepared_data, feature_config, training_config)
        st.success("Training abgeschlossen.")

    if not st.session_state.model_training:
        return

    metric_cols = st.columns(6)
    metrics = [
        ("RMSE", f"{st.session_state.rmse:.0f}"),
        ("CV RMSE", f"{st.session_state.cv_summary.mean_rmse:.0f}"),
        ("MAE", f"{st.session_state.mae:.0f}"),
        ("CV MAE", f"{st.session_state.cv_summary.mean_mae:.0f}"),
        ("R2", f"{st.session_state.r2:.3f}"),
        ("CV R2", f"{st.session_state.cv_summary.mean_r2:.3f}"),
    ]
    for column, (label, value) in zip(metric_cols, metrics, strict=False):
        with column:
            st.metric(label, value)

    action_cols = st.columns([1.2, 1.2, 3])
    with action_cols[0]:
        if st.button("Training speichern"):
            _save_current_pipeline(prepared_data, feature_config, training_config)
    with action_cols[1]:
        if st.session_state.show_scenarios_popover:
            st.success("Run ist im Session-Log sichtbar.")

    with st.expander("Pipeline-Struktur", expanded=False):
        components.html(
            estimator_html_repr(st.session_state.pipe),
            height=300,
            scrolling=True,
        )


def _render_diagnostics_tab(
    prepared_data,
    feature_config: FeatureSelectionConfig,
    training_config: TrainingConfig,
) -> None:
    if not st.session_state.model_training:
        st.info("Trainiere zuerst ein Modell, um Diagnostik und Fehlersichten zu sehen.")
        return

    diag_tabs = st.tabs(["Gesamt", "Residuen", "Feature Importance", "Baselines"])
    with diag_tabs[0]:
        st.plotly_chart(
            build_prediction_scatter(
                st.session_state.evaluation_report.y_true,
                st.session_state.evaluation_report.y_pred,
            ),
            use_container_width=True,
        )

    with diag_tabs[1]:
        _render_residual_analysis_tab()

    with diag_tabs[2]:
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

    with diag_tabs[3]:
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
        st.dataframe(
            baseline_df.style.format({"RMSE": "{:.0f}", "R2": "{:.3f}", "MAE": "{:.0f}"}),
            hide_index=True,
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
            st.caption("Gespeicherte Vorhersagen")
            st.dataframe(st.session_state.car_price.tail(10), use_container_width=True)


def _render_artifacts_tab() -> None:
    if st.session_state.scenario_log.empty:
        st.info("Noch keine gespeicherten Runs vorhanden.")
    else:
        st.markdown("**Gespeicherte Runs**")
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
            hide_index=True,
        )

    render_download_section(
        st.session_state.scenario_log,
        st.session_state.car_price,
        st.session_state.pipeline_store,
    )


def render_page() -> None:
    initialize_modeling_state()
    if consume_pending_screening_candidate():
        st.rerun()

    st.title(":material/model_training: Machine Learning zur Preisvorhersage")
    st.divider()
    st.caption(
        "Die Seite folgt dem Workflow "
        "Setup -> Vergleich -> Training -> Diagnose -> Vorhersage -> Artefakte."
    )

    base_df = _load_modeling_base_df()
    feature_options = [feature for feature in base_df.columns.tolist() if feature != "price"]
    engineered_feature_options = get_engineered_feature_names()
    page_tabs = st.tabs(
        [
            "Setup",
            "Kandidaten-Screening",
            "Finales Training",
            "Diagnose",
            "Vorhersage",
            "Artefakte",
        ]
    )

    with page_tabs[0]:
        st.subheader("Setup")
        candidate_locked = st.session_state.setup_candidate_label is not None
        if candidate_locked:
            info_cols = st.columns([3, 1])
            with info_cols[0]:
                st.info(
                    "Ein Kandidat aus dem Screening ist aktiv. "
                    "Modell, Skalierer und PCA sind aus dem Screening übernommen."
                )
            with info_cols[1]:
                if st.button("Kandidat lösen"):
                    clear_screening_candidate()
                    st.rerun()

        feature_cols = st.columns([1.5, 1.5, 2])
        with feature_cols[0]:
            base_features = st.multiselect(
                "Basis-Features",
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

        config_cols = st.columns([1.4, 1.4, 1.2, 1.2, 1.4])
        with config_cols[0]:
            model_selection = st.selectbox(
                "Regressionsmodell",
                ("lr", "rf", "xgb", "cat", "lgb"),
                format_func=lambda key: MODEL_LABELS[key],
                key="setup_model_key",
                disabled=candidate_locked,
            )
        with config_cols[1]:
            if model_selection == "lr":
                scaler_selection = st.segmented_control(
                    "Skalierer",
                    options=["None", "Standard", "MinMax"],
                    default=st.session_state.setup_scaler_key,
                    key="setup_scaler_key",
                    disabled=candidate_locked,
                )
            else:
                scaler_selection = "None"
                st.badge("Skalierung nicht erforderlich", color="green")
        with config_cols[2]:
            target_transform = st.selectbox(
                "Target",
                options=["raw", "log1p"],
                format_func=lambda key: {"raw": "Preis direkt", "log1p": "log1p(Preis)"}[key],
                key="setup_target_transform",
            )
            test_size = st.slider(
                "Validierungsanteil",
                10,
                50,
                st.session_state.setup_test_size_pct,
                5,
                key="setup_test_size_pct",
            ) / 100
        with config_cols[3]:
            cv_folds = st.slider(
                "CV Folds",
                3,
                5,
                st.session_state.setup_cv_folds,
                1,
                key="setup_cv_folds",
            )
            pca_check = model_selection == "lr" and st.checkbox(
                "PCA aktivieren?",
                value=st.session_state.setup_pca_enabled,
                key="setup_pca_enabled",
                disabled=candidate_locked,
            )
        with config_cols[4]:
            n_components = st.session_state.setup_n_components

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

        with st.expander("Verwendete Features", expanded=False):
            render_selected_feature_lists(
                prepared_data.categorical_columns,
                prepared_data.numeric_columns,
            )

        render_training_summary(
            prepared_data.x_train,
            prepared_data.x_test,
            len(feature_config.all_features),
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
                    "PCA-Komponenten",
                    min_value=1,
                    max_value=max_components,
                    value=min(n_opt, max_components),
                    key="setup_n_components",
                    disabled=candidate_locked,
                )
                st.caption(f"95% erklärte Varianz bei ca. {n_opt} Komponenten.")
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
        if st.session_state.setup_candidate_label:
            st.caption(f"Setup aus Screening übernommen: {st.session_state.setup_candidate_label}")

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

    if not screening_candidate_still_matches(
        model_key=training_config.model_key,
        scaler_key=training_config.scaler_key,
        pca_enabled=training_config.pca_enabled,
        n_components=training_config.n_components,
    ):
        clear_screening_candidate()

    with page_tabs[1]:
        _render_model_comparison(prepared_data, training_config)
    with page_tabs[2]:
        _render_training_tab(prepared_data, feature_config, training_config)
    with page_tabs[3]:
        _render_diagnostics_tab(prepared_data, feature_config, training_config)
    with page_tabs[4]:
        _render_prediction_section(prepared_data, feature_config)
    with page_tabs[5]:
        _render_artifacts_tab()
