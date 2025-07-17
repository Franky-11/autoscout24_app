import joblib
import io

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.utils import estimator_html_repr

import streamlit as st
import streamlit.components.v1 as components

from ml_functions import*


if 'model_training' not in st.session_state:
    st.session_state.model_training=False


if 'scenario_log' not in st.session_state:
        st.session_state.scenario_log = pd.DataFrame(columns=[
            'Pipe ID','Features','Anzahl Features', 'Model', 'Scaler', 'PCA Active', 'PCA Components','Test Size',
            'Model Parameters', 'R2 Score', 'RMSE','MAE'])


if 'car_price' not in st.session_state:
        st.session_state.car_price = pd.DataFrame(columns=[
            'Pipe ID','Marke','Modell','PS','Erstzulassung','Kilometerstand','Kraftstoff','Getriebe','Vorhergesagter Preis'])



if 'show_scenarios_popover' not in st.session_state:
    st.session_state.show_scenarios_popover = False



if 'features' not in st.session_state:
    st.session_state.features = None
if 'model' not in st.session_state:
    st.session_state.model = ''
if 'scaler' not in st.session_state:
    st.session_state.scaler = ''
if 'pca_check' not in st.session_state:
    st.session_state.pca_check = False

if 'n_components' not in st.session_state:
    st.session_state.n_components = 0

if 'test_size' not in st.session_state:
    st.session_state.test_size = 0

if 'pipeline_store' not in st.session_state:
    st.session_state.pipeline_store = {}


df_2=read_preprocess_df()



st.title(":material/model_training: Machine Learning zur Preisvorhersage")

st.divider()

st.subheader("🛠️ Feature Auswahl")
st.write("")
options=[feature for feature in df_2.columns.tolist() if feature!="price" ]

cols=st.columns(4)
with cols[0]:
    features=st.multiselect("Features",options=options,default=['hp','fuel','gear','make','mileage','year'])


cat_cols,num_cols=cat_num_cols(df_2,features)
df_features=feature_df(df_2,features,cat_cols)

categorical_dummy_cols = [col for col in df_features.columns if col not in num_cols]
X = df_features.copy()
y = df_2['price']


with cols[2]:
    st.markdown("**Ausegewählte kategorische Features**")
    for feature in cat_cols:
        st.markdown(f"* **{feature}**")
with cols[3]:
    st.markdown("**Ausegewählte numerische Features**")
    for feature in num_cols:
        st.markdown(f"* **{feature}**")

with cols[0]:
    st.write("")
    st.metric("Anzahl Features (nach One-Hot-Encoding)",value=df_features.shape[1],border=False)


#---------------------------------------------------------------------------------------#

st.divider()
st.subheader("⚙️ Modell Auswahl und Preprocessing")
st.write("")
cols=st.columns([1,0.5,2.5,0.5,2])
with cols[0]:
    model_selection = st.selectbox(
        'Wähle ein Regressionsmodell:',
        ('lr', 'rf', 'xgb'),
        format_func=lambda x: {'lr': 'Linear Regression', 'rf': 'Random Forest', 'xgb': 'XGBoost'}[x])

with cols[0]:
    st.write("")
    scaler_selection = st.segmented_control( 'Wähle einen Skalierer für numerische Features:',options=['None', 'Standard', 'MinMax'],default='Standard')
    st.write("")
    test_size = st.slider("Anteil Validierungsset", min_value=10, max_value=50, value=20, step=5) / 100


with cols[2]:
    n_components = 10
    pca_check = st.checkbox('PCA aktivieren?', value=False)


model_dict={'lr':'Linear Regression','rf':'Random Forest','xgb':'XGBoost'}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

explained_variance=pca_explained_variance(X_train,model_selection,scaler_selection,categorical_dummy_cols,num_cols)



with cols[2]:
    if pca_check:
        fig = plot_explained_var(explained_variance)
        #Anzahl PC für 95% Varianz
        n_opt = np.argmax(explained_variance >= 0.95) + 1

        n_components = st.slider('Anzahl der PCA-Komponenten:', min_value=1,
                                     max_value=len(num_cols) + len(categorical_dummy_cols) - 1, value=n_opt)
        st.write(f"▶️ Optimale PCA-Komponenten für 95% erkärte Varianz: {n_opt}")
        st.plotly_chart(fig)


with cols[4]:
    st.markdown("**Ausegewählte Parameter**")
    st.markdown(f"* **Modell**: {model_dict[model_selection]}")
    st.markdown(f"* **Skalierer**: {scaler_selection}")
    st.markdown(f"* **PCA**: {"Aktiviert" if pca_check else "ohne PCA"} {"mit" if pca_check else ""} {n_components if pca_check else ""} {"Komponenten" if pca_check else ""}")




parameters_changed = (
    features != st.session_state.features or
    model_selection != st.session_state.model or
    scaler_selection != st.session_state.scaler or
    pca_check != st.session_state.pca_check or
    (pca_check and n_components != st.session_state.n_components) or
    test_size != st.session_state.test_size
)


if parameters_changed:
    st.session_state.model_training = False
    st.session_state.show_scenarios_popover = False #

st.session_state.features = features
st.session_state.model = model_selection
st.session_state.scaler = scaler_selection
st.session_state.pca_check = pca_check
st.session_state.n_components = n_components
st.session_state.test_size = test_size

#---------------------------------------------------------------------------------------#

st.divider()
st.subheader("📈 Modell trainieren und evaluieren")
st.write("")
cols=st.columns(2)
with cols[0]:
    st.metric("Anzahl Trainingsdaten", value=X_train.shape[0])
    st.write("")
    st.metric("Anzahl Validierungsdaten", value=X_test.shape[0])


with cols[1]:
    placeholder = st.empty()
    if not st.session_state.model_training:
        placeholder.info("Modell noch nicht trainiert!")
    if st.button("Modell trainieren"):
        st.session_state.model_training = True
        st.session_state.show_scenarios_popover = False


        model,scaler=model_and_scaler(model_selection,scaler_selection)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, num_cols),
                ('cat', 'passthrough', categorical_dummy_cols)
            ])
        pipe=ml_pipe(preprocessor,pca_check,n_components,model_selection,model)

        st.markdown("**Struktur der Pipeline:**")

        pipeline_html = estimator_html_repr(pipe)

        components.html(pipeline_html, height=300, scrolling=True)

        with st.spinner("Training läuft... Dies kann einen Moment dauern."):
            # Modell trainieren
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            rmse, r_2, mae=performance(y_test,y_pred)

            st.session_state.y_pred = y_pred
            st.session_state.rmse = rmse
            st.session_state.r2 = r_2
            st.session_state.mae = mae
            st.session_state.pipe = pipe
            st.session_state.model_obj = model



        placeholder.success('Training abgeschlossen!')


if st.session_state.model_training:
    with st.container(border=True,height=900):
        cols=st.columns([3,0.5,1,0.5])
        with cols[0]:
            st.markdown(f"**Modell Performance {model_dict[model_selection]} auf dem Validierungsset**")
            inner_cols=st.columns(3)
            with inner_cols[0]:
                st.metric("RMSE",value=f"{st.session_state.rmse:.0f}")
            with inner_cols[1]:
                st.metric("R2",value=f"{st.session_state.r2:.3f}")
            with inner_cols[2]:
                st.metric("MAE", value=f"{st.session_state.mae:.0f}")


            qq=qq_plot(y_test,st.session_state.y_pred)
            res_skew,res_kurt,delta_skew,delta_kurt=check_residuals(y_test, st.session_state.y_pred)

            with st.popover("Residuen Analyse",use_container_width=True):
                tab_qq,tab_error=st.tabs(["QQ-Plot","Fehler nach Preissegmenten"])
                with tab_qq:
                    st.markdown("**Verteilungskennzahlen**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Schiefe (Skewness)",
                            value=f"{res_skew:.2f}",
                            delta=f"{delta_skew:.2f} zur Normalform",
                            delta_color="inverse" if delta_skew < 0.5 else "off"
                        )
                    with col2:
                        st.metric(
                            label="Spitzigkeit (Kurtosis)",
                            value=f"{res_kurt:.2f}",
                            delta=f"{delta_kurt:.2f} zur Normalform",
                            delta_color="inverse" if delta_kurt < 0.5 else "off"
                        )
                    st.divider()
                    st.markdown("**QQ-Plot der Residuen**")
                    st.plotly_chart(qq,use_container_width=True)


                with tab_error:
                    st.markdown("**Fehler nach Preissegmenten**")
                    st.plotly_chart(error_price_segments(y_test,st.session_state.y_pred))


            fig = plot_pred_vs_true(y_test, st.session_state.y_pred)
            st.plotly_chart(fig, use_container_width=True)

        with cols[2]:
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.markdown("**Feature Importance**")
            df_importance=feature_importance(st.session_state.pipe,model_selection)
            st.dataframe(df_importance,use_container_width=True,hide_index=True)



        if st.button("Pipeline speichern"):
            new_scenario_id = len(st.session_state.scenario_log) + 1
            model_params = st.session_state.model_obj.get_params()
            relevant_params = {k: v for k, v in model_params.items() if
                              not k.startswith(('regressor__', 'alpha', 'random_state', 'n_jobs'))}

            new_row = {
                'Pipe ID': new_scenario_id,
                'Features': features,
                'Anzahl Features':len(df_features.columns),
                'Model': model_dict[model_selection],
                'Scaler': scaler_selection,
                'PCA Active': pca_check,
                'PCA Components': n_components if pca_check else 'N/A',
                'Test Size': test_size,
                'Model Parameters': str(relevant_params),
                'R2 Score': st.session_state.r2,
                'RMSE': st.session_state.rmse,
                'MAE':st.session_state.mae
            }

            st.session_state.scenario_log = pd.concat([st.session_state.scenario_log, pd.DataFrame([new_row])],
                                                      ignore_index=True)
            st.session_state.pipeline_store[new_scenario_id] = st.session_state.pipe

            st.session_state.show_scenarios_popover = True
            st.success("Pipeline gespeichert!")

        if st.session_state.show_scenarios_popover:
            with st.popover("Gespeicherte Pipelines",use_container_width=True):
                st.dataframe(st.session_state.scenario_log)



#---------------------------------------------------------------------------------------#


st.divider()

if st.session_state.model_training:
    st.subheader("💸 Preis vorhersagen")
    st.write("")
    pca_text = "PCA aktiviert mit {} Komponenten".format(
        st.session_state.n_components) if st.session_state.pca_check else "PCA deaktiviert"

    st.markdown(f"""
    **Pipeline-Konfiguration für Vorhersage**
    -    Skalierer: **{st.session_state.scaler}**
    -    {pca_text}
    -    Modell: **{model_dict[st.session_state.model]}**
    """)



    hp=None
    year=None
    mileage=None
    make=None
    fuel=None
    gear=None
    model=None

    st.subheader("")

    with st.container(border=True,height=600):
        cols = st.columns(3)

        with cols[0]:
          if 'hp' in features:
             hp=st.slider("PS",min_value=int(df_2['hp'].min()),max_value=int(df_2['hp'].max()),value=135,step=5)
          if 'year' in features:
              year=st.number_input("Erstzulassung",2011,2021,2015,step=1)
          if 'mileage' in features:
              mileage=st.slider("Kilometerstand",min_value=0,max_value=df_2['mileage'].max(),value=15000,step=1000)

        with cols[1]:
           if 'make' in features:
               make = st.selectbox("Marke", sorted(df_2['make'].unique()), index=5)
           if 'model' in features:
               model_options = sorted(
                    df_2[df_2['make'] == make]['model'].unique()) if 'make' in features else sorted(
                    df_2['model'].unique())
               model = st.selectbox("Modell", model_options, index=0)




        with cols[2]:
          if 'fuel' in features:
              fuel=st.segmented_control("Kraftstoff",options=sorted(df_2['fuel'].unique()),default='Gasoline')
          if 'gear' in features:
              gear=st.segmented_control("Getriebe",options=sorted(df_2['gear'].unique()),default='Manual')


        new_car_df=car_data(categorical_dummy_cols,X_train,hp=hp ,year=year ,mileage=mileage ,make=make ,fuel=fuel ,gear=gear ,model=model)

        car_price=st.session_state.pipe.predict(new_car_df)


        new_price = {
            'Pipe ID':st.session_state.scenario_log['Pipe ID'].iloc[-1] if not st.session_state.scenario_log.empty else 'N/A',
            'Marke': make,
            'Modell': model,
            'PS': hp,
            'Erstzulassung': year,
            'Kilometerstand': mileage,
            'Kraftstoff': fuel,
            'Getriebe': gear,
            'Vorhergesagter Preis': f"{car_price[0]:.0f}"
          }

        st.markdown("-------------")
        col_predict,col_save,col_pop=st.columns([0.5,1,2])

        with col_predict:
            st.metric('Vorhergesagter Preis', value=f"{car_price[0]:.0f}")

        with col_save:
            save_info=st.empty()
            if st.session_state.scenario_log.empty:
                save_info.info("Zum Speichern der Vorhersage Pipeline speichern!")
            elif not st.session_state.show_scenarios_popover:
                save_info.info("Neue Pipeline noch nicht gespeichert!")
            else:
                if st.button("Preisvorhersage speichern"):
                    st.session_state.car_price = pd.concat([st.session_state.car_price, pd.DataFrame([new_price])],                                        ignore_index=True)
                    st.success("Preisvorhersage gespeichert!")

        with col_pop:
            with st.popover("Gespeicherte Vorhersagen",use_container_width=True):
                st.dataframe(st.session_state.car_price,use_container_width=True)


st.divider()

##-----Download Sektion---"
with st.container(border=True):
    st.subheader("📦 Datenexport")
    st.markdown("Daten herunterladen:")
    col1, col2,col3 = st.columns(3)
    with col1:
        scenario_csv = st.session_state.scenario_log.to_csv(index=False)
        st.download_button(
            label="📥 Gespeicherte Pipelines als CSV",
            data=scenario_csv,
            file_name="pipe_log.csv",
            mime="text/csv"
        )

    with col2:
        car_price_csv = st.session_state.car_price.to_csv(index=False)
        st.download_button(
            label="📥 Preisvorhersagen als CSV",
            data=car_price_csv,
            file_name="predicted_prices.csv",
            mime="text/csv"
        )


    with col3:
        with st.popover("📥 Pipeline als .pkl herunterladen", use_container_width=True):
            st.markdown("Gespeicherte Pipeline auswählen:")
            if not st.session_state.pipeline_store:
                st.info("Keine gespeicherten Pipelines vorhanden.")
            else:
                pipe_ids = st.session_state.scenario_log["Pipe ID"].tolist()
                selected_id = st.selectbox("🔢 Pipe ID", pipe_ids)

                selected_row = st.session_state.scenario_log[st.session_state.scenario_log["Pipe ID"] == selected_id]
                st.markdown(
                    f"**Modell:** {selected_row['Model'].values[0]}  \n**Scaler:** {selected_row['Scaler'].values[0]}  \n**PCA:** {selected_row['PCA Active'].values[0]}  \n**RMSE:** {selected_row['RMSE'].values[0]:.0f}")

                if selected_id in st.session_state.pipeline_store:
                    pipe_obj = st.session_state.pipeline_store[selected_id]
                    buffer = io.BytesIO()
                    joblib.dump(pipe_obj, buffer)
                    buffer.seek(0)

                    st.download_button(
                        label="📥 Ausgewählte Pipeline als .pkl herunterladen",
                        data=buffer,
                        file_name=f"pipeline_{selected_id}.pkl",
                        mime="application/octet-stream"
                    )
                else:
                    st.warning("❌ Keine Pipeline unter dieser ID gespeichert.")

