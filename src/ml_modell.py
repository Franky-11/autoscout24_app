import joblib
import io
import time

from sklearn.model_selection import train_test_split

from sklearn.utils import estimator_html_repr


import streamlit.components.v1 as components

from ml_functions import*


if 'model_training' not in st.session_state:
    st.session_state.model_training=False


if 'scenario_log' not in st.session_state:
        st.session_state.scenario_log = pd.DataFrame(columns=[
            'Pipe ID','Features','Anzahl Features', 'Model', 'Scaler', 'PCA Active', 'PCA Components','Test Size',
            'Model Parameters','Train Time (s)', 'R2 Score', 'RMSE','MAE'])


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



st.title(":material/model_training: Machine Learning zur Preisvorhersage")

st.divider()

st.subheader("🛠️ Feature Auswahl")
st.write("")

df_2=read_preprocess_df()

df,engineered_features=feature_engineering(df_2)

options=[feature for feature in df_2.columns.tolist() if feature!="price" ]
options_engineered=engineered_features

cols=st.columns(4)
with cols[0]:
    features=st.multiselect("Features",options=options,default=['hp','fuel','gear','make','mileage','year'])

with cols[1]:
    engineered_features=st.multiselect("Engineered Features",options=options_engineered)

features=features+engineered_features

cat_cols,num_cols=cat_num_cols(df,features)


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
   # st.metric("Anzahl Features (nach One-Hot-Encoding)",value=df_features.shape[1],border=False)
#   st.write(df_features.isna().sum())

#---------------------------------------------------------------------------------------#

st.divider()
st.subheader("⚙️ Modell Auswahl und Preprocessing")
st.write("")
cols=st.columns([1.5,0.5,2.5,0.5,2])
with cols[0]:
    model_selection = st.selectbox(
        'Wähle ein Regressionsmodell:',
        ('lr', 'rf', 'xgb','cat','lgb'),
        format_func=lambda x: {'lr': 'Linear Regression', 'rf': 'Random Forest', 'xgb': 'XGBoost','cat':'CatBoost','lgb':'LightGBM'}[x])

with cols[0]:
    if model_selection=="lr":
        scaler_selection = st.segmented_control( 'Wähle einen Skalierer für numerische Features:',options=['None', 'Standard', 'MinMax'],default='Standard')
        st.badge("One-Hot-Encoding wird verwendet", color="blue")
    elif model_selection in ['xgb','rf']:
        scaler_selection='None'
        st.badge("Skalierung nicht erforderlich", color="green")
        st.badge("One-Hot-Encoding wird verwendet", color="blue")
    else:
        scaler_selection = 'None'
        st.badge("Skalierung nicht erforderlich", color="green")
        st.badge(" Kein One-Hot-Encoding", color="green")

    st.write("")
    test_size = st.slider("Anteil Validierungsset", min_value=10, max_value=50, value=20, step=5) / 100


with cols[2]:
    n_components = 10
    if model_selection == "lr":
        pca_check = st.checkbox('PCA aktivieren?', value=False)
    else:
        pca_check = False


df_features=feature_df(df,features,cat_cols,model_selection)

categorical_dummy_cols = [col for col in df_features.columns if col not in num_cols]
X = df_features.copy()
y = df['price']


model_dict={'lr':'Linear Regression','rf':'Random Forest','xgb':'XGBoost','cat':'CatBoost','lgb':'LightGBM'}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

explained_variance=pca_explained_variance(X_train,model_selection,scaler_selection,categorical_dummy_cols,num_cols,cat_cols)


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

current_config = {
    "Features": sorted(features),
    "Model": model_dict[model_selection],
    "Scaler": scaler_selection,
    "PCA Active": pca_check,
    "PCA Components": n_components if pca_check else 'N/A',
    "Test Size": test_size}


exists = False
for idx, row in st.session_state.scenario_log.iterrows():
    row_config = {
        "Features": sorted(row["Features"]),
        "Model": row["Model"],
        "Scaler": row["Scaler"],
        "PCA Active": row["PCA Active"],
        "PCA Components": row["PCA Components"],
        "Test Size": row["Test Size"]
    }
    if current_config == row_config:
        exists = True
        break


if parameters_changed and not exists:
#if not exists:
    st.session_state.model_training = False
    st.session_state.show_scenarios_popover = False

else:
    st.session_state.model_training = True
    st.session_state.show_scenarios_popover = True




#---------------------------------------------------------------------------------------#

st.divider()
st.subheader("📈 Modell trainieren und evaluieren")
st.write("")
cols=st.columns([1,1,2])
with cols[0]:
    st.metric("Anzahl Trainingsdaten", value=X_train.shape[0])
    st.write("")
    st.metric("Anzahl Validierungsdaten", value=X_test.shape[0])

with cols[1]:
    text = "Anzahl Features" if len(features)==X_train.shape[1] else "Anzahl Features mit One-Hot-Encoding"
    st.metric(text, value=X_train.shape[1])


with cols[2]:
    placeholder = st.empty()
    if exists:
        placeholder.info("Modellsetup bereits trainiert und abgespeichert!")
    if not st.session_state.model_training:
        placeholder.info("Modell noch nicht trainiert!")

    if st.button("Modell trainieren"):
        st.session_state.model_training = True
        st.session_state.show_scenarios_popover = False


        model,scaler=model_and_scaler(model_selection,scaler_selection,cat_cols)

        pipe=ml_pipe(pca_check,n_components,model_selection,model,scaler,categorical_dummy_cols,num_cols)

        st.markdown("**Struktur der Pipeline:**")

        pipeline_html = estimator_html_repr(pipe)

        components.html(pipeline_html, height=300, scrolling=True)

        with st.spinner("Training läuft... Dies kann einen Moment dauern."):
            # Modell trainieren
            if model_selection=='lgb':
                start = time.perf_counter()
                pipe.fit(X_train,y_train,lgb__categorical_feature=cat_cols)
                delta = time.perf_counter() - start

            else:
                start = time.perf_counter()
                pipe.fit(X_train, y_train)
                delta = time.perf_counter() - start


            y_pred = pipe.predict(X_test)
            rmse, r_2, mae=performance(y_test,y_pred)

            st.session_state.train_time = delta
            st.session_state.X_train = X_train
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.rmse = rmse
            st.session_state.r2 = r_2
            st.session_state.mae = mae
            st.session_state.pipe = pipe
            st.session_state.model_obj = model
            st.session_state.model_selection = model_selection
            st.session_state.categorical_dummy_cols = categorical_dummy_cols

            st.session_state.features = features
            st.session_state.model = model_selection
            st.session_state.scaler = scaler_selection
            st.session_state.pca_check = pca_check
            st.session_state.n_components = n_components
            st.session_state.test_size = test_size





        placeholder.success('Training abgeschlossen!')


if st.session_state.model_training:
    with st.container(border=True,height=1000):
        cols=st.columns([3,0.5,1,0.5])
        with cols[0]:
            st.markdown(f"**Modell Performance {model_dict[st.session_state.model_selection]} auf dem Validierungsset**")
            inner_cols=st.columns(3)
            with inner_cols[0]:
                st.metric("RMSE",value=f"{st.session_state.rmse:.0f}")
            with inner_cols[1]:
                st.metric("R2",value=f"{st.session_state.r2:.3f}")
            with inner_cols[2]:
                st.metric("MAE", value=f"{st.session_state.mae:.0f}")


            qq=qq_plot(st.session_state.y_test,st.session_state.y_pred)
            res_skew,res_kurt,delta_skew,delta_kurt=check_residuals(st.session_state.y_test, st.session_state.y_pred)

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
                    st.plotly_chart(error_price_segments(st.session_state.y_test,st.session_state.y_pred))


            fig = plot_pred_vs_true(st.session_state.y_test, st.session_state.y_pred)
            st.plotly_chart(fig, use_container_width=True)

        with cols[2]:
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.markdown("**Feature Importance**")
            df_importance=feature_importance(st.session_state.pipe,st.session_state.model_selection,st.session_state.X_train)
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
                'Train Time (s)': st.session_state.train_time,
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
                st.dataframe(st.session_state.scenario_log.style.format({
                    'Test Size': "{:.2f}",
                    'Train Time (s)': "{:.2f}",
                    'R2 Score': "{:.3f}",
                    'RMSE': "{:.0f}",
                    'MAE': "{:.0f}"
                }))

            if not st.session_state.scenario_log.empty:
                best_id=st.session_state.scenario_log["RMSE"].idxmin()
                best_pipeline=st.session_state.scenario_log.loc[best_id]
                st.markdown(
                    f"""**Beste Pipeline bisher ➡️**&nbsp;&nbsp;&nbsp;&nbsp; 
                      **Pipe ID**&nbsp;&nbsp; {best_pipeline['Pipe ID']}&nbsp;&nbsp; |&nbsp;&nbsp; **Modell**&nbsp;&nbsp; {best_pipeline['Model']}&nbsp;&nbsp; |&nbsp;&nbsp; **RMSE**&nbsp;&nbsp; {int(best_pipeline['RMSE'])}
                """)





#---------------------------------------------------------------------------------------#


st.divider()

if st.session_state.model_training:
    st.subheader("💸 Preis vorhersagen")
    st.write("")
    pca_text = "PCA aktiviert mit {} Komponenten".format(
        st.session_state.n_components) if st.session_state.pca_check else "PCA deaktiviert"

    save_info = st.empty()
    if st.session_state.scenario_log.empty:
        save_info.info("Zum Speichern der Vorhersage Pipeline speichern!")
    elif not st.session_state.show_scenarios_popover:
        save_info.info("Neue Pipeline noch nicht gespeichert!")
    else:
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

    required_widgets=input_features(st.session_state.features)

    st.subheader("")

    with st.container(border=True,height=600):
        cols = st.columns(3)

        with cols[0]:
          if 'hp' in required_widgets:
             hp=st.slider("PS",min_value=int(df['hp'].min()),max_value=int(df['hp'].max()),value=135,step=5)
          if 'year' in required_widgets:
              year=st.number_input("Erstzulassung",2011,2021,2015,step=1)
          if 'mileage' in required_widgets:
              mileage=st.slider("Kilometerstand",min_value=0,max_value=df['mileage'].max(),value=15000,step=1000)

        with cols[1]:
           if 'make' in required_widgets:
               make = st.selectbox("Marke", sorted(df['make'].unique()), index=5)
           if 'model' in required_widgets:
               model_options = sorted(
                    df[df['make'] == make]['model'].unique()) if 'make' in features else sorted(
                    df['model'].unique())
               model = st.selectbox("Modell", model_options, index=0)


        with cols[2]:
          if 'fuel' in required_widgets:
              fuel=st.segmented_control("Kraftstoff",options=sorted(df['fuel'].unique()),default='Gasoline')
          if 'gear' in required_widgets:
              gear=st.segmented_control("Getriebe",options=sorted(df['gear'].unique()),default='Manual')


        new_car_df=car_data(st.session_state.model_selection,st.session_state.categorical_dummy_cols,st.session_state.X_train,cat_cols,hp=hp ,year=year ,mileage=mileage ,make=make ,fuel=fuel ,gear=gear ,model=model)


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
            if not st.session_state.scenario_log.empty and st.session_state.show_scenarios_popover:
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

