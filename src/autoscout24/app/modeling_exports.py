import io

import joblib
import pandas as pd
import streamlit as st


def render_download_section(
    scenario_log: pd.DataFrame,
    prediction_log: pd.DataFrame,
    pipeline_store: dict[int, object],
) -> None:
    with st.container(border=True):
        st.subheader("📦 Datenexport")
        st.markdown("Daten herunterladen:")
        col1, col2, col3 = st.columns(3)

        with col1:
            scenario_csv = scenario_log.to_csv(index=False)
            st.download_button(
                label="📥 Gespeicherte Pipelines als CSV",
                data=scenario_csv,
                file_name="pipe_log.csv",
                mime="text/csv",
            )

        with col2:
            prediction_csv = prediction_log.to_csv(index=False)
            st.download_button(
                label="📥 Preisvorhersagen als CSV",
                data=prediction_csv,
                file_name="predicted_prices.csv",
                mime="text/csv",
            )

        with col3:
            with st.popover("📥 Pipeline als .pkl herunterladen", use_container_width=True):
                st.markdown("Gespeicherte Pipeline auswählen:")
                if not pipeline_store:
                    st.info("Keine gespeicherten Pipelines vorhanden.")
                else:
                    pipe_ids = scenario_log["Pipe ID"].tolist()
                    selected_id = st.selectbox("🔢 Pipe ID", pipe_ids)
                    selected_row = scenario_log[scenario_log["Pipe ID"] == selected_id]
                    st.markdown(
                        f"**Modell:** {selected_row['Model'].values[0]}  \n"
                        f"**Scaler:** {selected_row['Scaler'].values[0]}  \n"
                        f"**Target:** {selected_row['Target'].values[0]}  \n"
                        f"**RMSE:** {selected_row['RMSE'].values[0]:.0f}"
                    )

                    if selected_id in pipeline_store:
                        pipe_obj = pipeline_store[selected_id]
                        buffer = io.BytesIO()
                        joblib.dump(pipe_obj, buffer)
                        buffer.seek(0)

                        st.download_button(
                            label="📥 Ausgewählte Pipeline als .pkl herunterladen",
                            data=buffer,
                            file_name=f"pipeline_{selected_id}.pkl",
                            mime="application/octet-stream",
                        )
                    else:
                        st.warning("❌ Keine Pipeline unter dieser ID gespeichert.")
