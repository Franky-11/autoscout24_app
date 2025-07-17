
#import os
from pathlib import Path


import numpy as np
import pandas as pd


import plotly.express as px
import plotly.graph_objects as go


from scipy.stats import kurtosis, probplot, skew


from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from xgboost import XGBRegressor



def read_preprocess_df(factor = 1.5):
    """
        Lädt und bereinigt die Autoscout24-Daten, entfernt Ausreißer via IQR,
        kategorisiert relevante Spalten und bereitet das finale Modell-DataFrame vor.

        Args:
            factor (float): Faktor zur Bestimmung der IQR-Grenzen (Default: 1.5).

        Returns:
            pd.DataFrame: Vorverarbeiteter und ausreißerbereinigter Datensatz.
    """
    script_path = Path(__file__).resolve()
    src_dir = script_path.parent
    data_file_path = src_dir / "autoscout24.csv"

    df = pd.read_csv(data_file_path)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    iqr_bounds = df.groupby("year").agg(
        # Aggregationen für 'price'
        price_q1=('price', lambda x: x.quantile(0.25)),
        price_q3=('price', lambda x: x.quantile(0.75)),
        # Aggregationen für 'mileage'
        mileage_q1=('mileage', lambda x: x.quantile(0.25)),
        mileage_q3=('mileage', lambda x: x.quantile(0.75))
    ).reset_index()
    iqr_bounds["price_u_limit"] = iqr_bounds["price_q3"] + (iqr_bounds["price_q3"] - iqr_bounds["price_q1"]) * factor
    iqr_bounds["mileage_u_limit"] = iqr_bounds["mileage_q3"] + (
                iqr_bounds["mileage_q3"] - iqr_bounds["mileage_q1"]) * factor

    df_merged = pd.merge(df, iqr_bounds[['year', 'price_u_limit', 'mileage_u_limit']], on='year', how='left')
    df_filtered_iqr = df_merged[
        (df_merged['price'] <= df_merged['price_u_limit']) & (df_merged['mileage'] <= df_merged['mileage_u_limit'])]

    df_2 = df_filtered_iqr.drop(['price_u_limit', 'mileage_u_limit', 'offerType'], axis=1)
    df_2['make'] = df_2['make'].astype('category')
    df_2['fuel'] = df_2['fuel'].astype('category')
    df_2['gear'] = df_2['gear'].astype('category')
    df_2['model'] = df_2['model'].astype('category')

    # Weitere Features erzeugen
   # df_2['mileage_square'] = df_2['mileage'] ** 2
   # df_2['year_square'] = df_2['year'] ** 2
   # df_2['ln_mileage'] = np.log(df_2['mileage'] + 1)
   # df_2['ln_year'] = np.log(df_2['year'])

    return df_2

def cat_num_cols(df_2,features):
    """
        Trennt Features in kategoriale und numerische Spalten basierend auf Datentyp.

        Args:
            df_2 (pd.DataFrame): Feature-DataFrame.
            features (list): Liste aller Feature-Namen.

        Returns:
            tuple: (List kategorischer Features, List numerischer Features)
    """
    categorical_cols = [feature for feature in features if df_2[feature].dtype == "category"]
    numerical_cols = [feature for feature in features if feature not in categorical_cols]
    return categorical_cols,numerical_cols



def feature_df(df_2,features,categorical_cols):
    """
        Wandelt kategoriale Variablen per One-Hot-Encoding in numerische Dummy-Features um.

        Args:
            df_2 (pd.DataFrame): Ursprünglicher Feature-DataFrame.
            features (list): Liste zu verwendender Spalten.
            categorical_cols (list): Liste der kategorialen Feature-Namen.

        Returns:
            pd.DataFrame: Modellfähiger Feature-Vektor.
    """
    df_features = df_2[features].copy()
    df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    return df_features



def model_and_scaler(model_selection,scaler_selection):
    """
        Gibt Modellobjekt und zugehörigen Skalierer zurück basierend auf Auswahl.

        Args:
            model_selection (str): Kürzel für Modelltyp ('lr', 'rf', 'xgb').
            scaler_selection (str): Skalierungsmethode ('Standard', 'MinMax', 'None').

        Returns:
            tuple: (Modellobjekt, Skalierungsobjekt oder 'passthrough')
    """


    model_dict={'lr':LinearRegression(),
                 'rf':RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_split=2,max_features='log2',random_state=42),
                'xgb':XGBRegressor(random_state=42, n_jobs=-1,colsample_bytree= 0.8, learning_rate= 0.1, max_depth=7, n_estimators= 300, subsample= 1.0)}


    scaler_dict={'Standard':StandardScaler(),'MinMax':MinMaxScaler(),'None':'passthrough'}

    model = model_dict[model_selection]
    scaler = scaler_dict[scaler_selection]

    return model,scaler

def pca_explained_variance(X_train,model_selection,scaler_selection,categorical_dummy_cols,num_cols):
    """
        Berechnet kumulierte erklärte Varianz aller Hauptkomponenten (PCA).

        Args:
            X_train (pd.DataFrame): Trainingsdaten.
            model_selection (str): Modelltyp zur Skalerauswahl.
            scaler_selection (str): Skalierungsmethode.
            categorical_dummy_cols (list): One-Hot-kodierte Features.
            num_cols (list): Numerische Feature-Namen.

        Returns:
            np.ndarray: Array kumulierter Varianz je Hauptkomponente.
    """

    _,scaler=model_and_scaler(model_selection,scaler_selection)

    preprocessor_for_pca = ColumnTransformer(
                transformers=[
                    ('num', scaler, num_cols),
                    ('cat', 'passthrough', categorical_dummy_cols)
                ])
    pipe_for_pca=Pipeline([('preprocessor', preprocessor_for_pca),('pca', PCA(n_components=X_train.shape[1]))])
    pipe_for_pca.fit_transform(X_train)
    explained_var=np.cumsum(pipe_for_pca['pca'].explained_variance_ratio_)

    return explained_var


def plot_explained_var(explained_var):
    """
        Erstellt ein Liniendiagramm zur Visualisierung kumulierter erklärter Varianz.

        Args:
            explained_var (np.ndarray): Varianzanteile der Hauptkomponenten.

        Returns:
            plotly.graph_objects.Figure: Interaktive Plotly-Figur.
    """
    vivid_colors = px.colors.qualitative.Vivid
    first_vivid_color = vivid_colors[0]
    df_var = pd.DataFrame({
        "Anzahl Komponenten": np.arange(1, len(explained_var) + 1),
        "Kumulierte erklärte Varianz": explained_var
    })

    fig = px.line(
        df_var,
        x="Anzahl Komponenten",
        y="Kumulierte erklärte Varianz",
        labels={
            "Kumulierte erklärte Varianz": "Kumulierte erklärte Varianz",
            "Anzahl Komponenten": "Anzahl der Komponenten"
        }
    )


    fig.update_traces(
        line_color=first_vivid_color,
        marker_color=first_vivid_color,
        mode='lines',
        line_width=3,
        marker_size=8
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=14),
            showgrid=True  #
        ),

        yaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=14),
            showgrid=True
        )
    )
    fig.update_layout(
        hovermode="x unified"
    )

    return fig


def ml_pipe(preprocessor,pca_check,n_components,model_selection,model):
    """
        Baut eine ML-Pipeline mit Preprocessing, optional PCA und gewähltem Modell.

        Args:
            preprocessor (ColumnTransformer): Feature-Transformation.
            pca_check (bool): Ob PCA integriert werden soll.
            n_components (int): Anzahl PCA-Komponenten.
            model_selection (str): Modellkürzel.
            model: Modellobjekt.

        Returns:
            Pipeline: Sklearn-Pipeline für Training und Vorhersage.
    """
    pipeline_steps = [('preprocessor', preprocessor)]
    if pca_check:
        pipeline_steps.append(('pca', PCA(n_components=n_components)))
    pipeline_steps.append((model_selection, model))
    pipe = Pipeline(pipeline_steps)
    return pipe


def performance(y_test,y_pred):
    """
        Berechnet klassische Regressionsmetriken.

        Args:
            y_test (pd.Series): Wahre Zielwerte.
            y_pred (np.ndarray): Modellvorhersagen.

        Returns:
            tuple: (RMSE, R²-Score, MAE)
    """
    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    r_2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    return rmse,r_2,mae


def plot_pred_vs_true(y_test,y_pred):
    """
        Visualisiert ein Scatterplot mit Vorhersage vs. Ist-Wert und Referenzlinie y = x.

        Returns:
            plotly.graph_objects.Figure: Vergleichsdiagramm.
    """

    fig = go.Figure()
    plot_color = px.colors.qualitative.Vivid[0]
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode="markers",
        marker=dict(
            color=plot_color,
            opacity=0.7
        ),
        name="Vorhersage vs. Ist-Wert"
    ))

    # --Referenzlinie für perfekte Vorhersage (y = x) ---
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='grey', dash='dash'),  # Gestrichelte graue Linie
        name='Perfekte Vorhersage (Y=X)'
    ))

    fig.update_layout(
        xaxis_title="Wahrer Preis",
        yaxis_title="Vorhergesagter Preis",
        height=555,
        hovermode="closest"
    )

    return fig


def qq_plot(y_test,y_pred):
    """
        Erstellt ein QQ-Plot zur Normalitätsprüfung der Residuen.

        Returns:
            plotly.graph_objects.Figure: QQ-Diagramm.
    """
    plot_color = px.colors.qualitative.Vivid[0]
    residuals=y_test-y_pred
    residuals_standard = (residuals - residuals.mean()) / residuals.std()

    qq_x, qq_y = probplot(residuals_standard, dist="norm", fit=False)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=qq_x, y=qq_y,
        mode='markers',
        name="Beobachtete Residuen-Quantile",
        marker=dict(
            color=plot_color,
            opacity=0.7
        ),
        hovertemplate="Theoretisch: %{x:.2f}<br>Beobachtet: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[min(qq_x), max(qq_x)],
        y=[min(qq_x), max(qq_x)],
        mode='lines',
        name="Normalverteilungs-Referenz",
        line=dict(color='grey', dash='dash'),
        hoverinfo='skip'
    ))

    fig.update_layout(
        xaxis = dict(
        title="Theoretische Quantile (Normalverteilung)",
        showgrid=False
    ),
        yaxis=dict(title="Beobachtete Quantile (Residuen)", showgrid=False))

    return fig



def error_price_segments(y_test,y_pred):
    """
        Visualisiert durchschnittliche Fehler und Residuen segmentiert nach Preisbereich.

        Returns:
            plotly.graph_objects.Figure: Balkendiagramm mit Residuenanalyse.
    """

    df_error = pd.DataFrame({
        "Istpreis": y_test,
        "Vorhersage": y_pred,
        "Residuum": y_test - y_pred,
        "Absoluter Fehler": np.abs(y_test - y_pred)
    })

    df_error["Preis-Segment"] = pd.cut(
        df_error["Istpreis"],
        bins=[0, 10000, 20000, 30000, 40000, 50000, np.inf],
        labels=["0–10k", "10k–20k", "20k–30k", "30k–40k", "40k–50k", "50k+"]
    )

    segment_stats = df_error.groupby("Preis-Segment").agg({
        "Absoluter Fehler": "mean",
        "Residuum": ["mean", "count"]
    }).reset_index()

    segment_stats.columns = ["Preis-Segment", "Ø Fehler", "Ø Residuum", "Anzahl"]

    fig = px.bar(
        segment_stats,
        x="Preis-Segment",
        y="Ø Residuum",
        color="Ø Residuum",
        color_continuous_scale="RdBu",
        text="Ø Residuum",
        custom_data=["Anzahl", "Ø Fehler"]
    )

    fig.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        hovertemplate=(
                "Segment: %{x}<br>" +
                "Ø Residuum: %{y:.0f} €<br>" +
                "Datenpunkte: %{customdata[0]}<br>" +
                "Ø Absoluter Fehler: %{customdata[1]:.0f} €<extra></extra>"
        )
    )

    fig.update_layout(
        yaxis_title="Ø Residuum",
        xaxis_title="Preisbereich (Istwerte)",
        height=450
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")

    max_val = max(abs(segment_stats["Ø Residuum"].min()), abs(segment_stats["Ø Residuum"].max()))
    fig.update_coloraxes(cmin=-max_val, cmax=max_val)

    return fig


def check_residuals(y_test,y_pred):
    """
        Berechnet Schiefe und Kurtosis der Residuen sowie deren Abweichung zur Normalform.

        Returns:
            tuple: (Skew, Kurtosis, ΔSkew zu 0, ΔKurtosis zu 3)
    """
    residuals = y_test - y_pred
    res_skew = skew(residuals)
    res_kurt = kurtosis(residuals, fisher=False)
    delta_skew = abs(res_skew - 0)  # Normalform = 0
    delta_kurt = abs(res_kurt - 3)  # Normalform = 3

    return res_skew,res_kurt,delta_skew,delta_kurt







def feature_importance(pipe, model_selection):
    """
        Gibt sortierte Feature-Importances basierend auf dem Modell zurück.
        Unterstützt sowohl Regressionskoeffizienten als auch Feature-Importances.

        Args:
            pipe (Pipeline): Trainierte Sklearn-Pipeline.
            model_selection (str): Modellkürzel ('lr', 'rf', 'xgb').

        Returns:
            pd.DataFrame: Feature-Importances mit Namen und Gewichtung.
    """
    model = pipe.named_steps[model_selection]

    if "pca" in pipe.named_steps:
        n_components = pipe.named_steps["pca"].n_components_
        feature_names = [f"PC{i + 1}" for i in range(n_components)]
    else:
        last_transformer = pipe.named_steps["preprocessor"]
        feature_names = last_transformer.get_feature_names_out()

    if model_selection == "lr":
        importance = model.coef_
    elif model_selection in ["rf", "xgb"]:
        importance = model.feature_importances_
    else:
        return None
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    return df_importance





def car_data(categorical_dummy_cols,X_train,hp=None,year=None,mileage=None,make=None,fuel=None,gear=None,model=None):
    """
        Erzeugt einen Feature-Vektor für ein einzelnes Fahrzeug basierend auf Benutzerangaben,
        passend zur Struktur des Trainingsdatensatzes (Dummy-kodierte Kategorien und numerische Merkmale).

        Args:
            categorical_dummy_cols (list): Liste aller Dummy-Features (One-Hot-kodierte Spaltennamen).
            X_train (pd.DataFrame): Trainingsdaten, um die Spaltenstruktur zu übernehmen.
            hp (float, optional): Motorleistung in PS.
            year (int, optional): Erstzulassungsjahr.
            mileage (float, optional): Kilometerstand.
            make (str, optional): Fahrzeugmarke.
            fuel (str, optional): Kraftstofftyp.
            gear (str, optional): Getriebeart.
            model (str, optional): Modellbezeichnung.

        Returns:
            pd.DataFrame: 1-zeiliger Feature-Vektor für Modellinput.
    """

    new_car_data = {}
    for col in categorical_dummy_cols:
        new_car_data[col] = 0

    if make is not None:
        make='make'+'_'+make
        new_car_data[make] = 1

    if fuel is not None:
        fuel='fuel'+'_'+fuel
        new_car_data[fuel] = 1

    if gear is not None:
        gear='gear'+'_'+gear
        new_car_data[gear] = 1

    if model is not None:
        model='model'+'_'+model
        new_car_data[model] = 1

    if hp is not None:
        new_car_data['hp'] = hp

    if year is not None:
        new_car_data['year'] = year
    if mileage is not None:
        new_car_data['mileage'] = mileage


    new_car_df = pd.DataFrame([new_car_data])
    new_car_df = new_car_df[X_train.columns]

    return new_car_df