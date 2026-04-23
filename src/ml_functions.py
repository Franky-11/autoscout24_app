import streamlit as st
import numpy as np
import pandas as pd


import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import kurtosis, probplot, skew

from autoscout24.data.io import load_modeling_dataset
from autoscout24.features.engineering import (
    apply_feature_engineering,
    build_feature_frame,
    build_vehicle_input_frame,
    get_engineered_feature_names,
    required_prediction_inputs,
    split_feature_types,
)
from autoscout24.modeling.service import (
    build_model_and_scaler,
    build_pipeline,
    calculate_pca_explained_variance,
    calculate_regression_metrics,
)


@st.cache_data
def read_preprocess_df(factor = 1.5):
    """
        Lädt und bereinigt die Autoscout24-Daten, entfernt Ausreißer via IQR,
        kategorisiert relevante Spalten und bereitet das finale Modell-DataFrame vor.

        Args:
            factor (float): Faktor zur Bestimmung der IQR-Grenzen (Default: 1.5).

        Returns:
            pd.DataFrame: Vorverarbeiteter und ausreißerbereinigter Datensatz.
    """
    return load_modeling_dataset(factor=factor)


def feature_engineering(df_2):
    """
    Erzeugt neue Features für die Preisvorhersage ohne Data Leakage.

    Args:
        df_2 (pd.DataFrame): Ursprünglicher Fahrzeugdatensatz bereinigt.

    Returns:
        pd.DataFrame: Erweiterter Datensatz mit neuen Features und engineered_feature Liste
    """
    return apply_feature_engineering(df_2), get_engineered_feature_names()



def cat_num_cols(df,features):
    """
        Trennt Features in kategoriale und numerische Spalten basierend auf Datentyp.

        Args:
            df (pd.DataFrame): Feature-DataFrame.
            features (list): Liste aller Feature-Namen.

        Returns:
            tuple: (List kategorischer Features, List numerischer Features)
    """
    return split_feature_types(df, features)



def feature_df(df,features,categorical_cols,model_selection):
    """
        Wandelt kategoriale Variablen per One-Hot-Encoding in numerische Dummy-Features um.

        Args:
            df (pd.DataFrame): Ursprünglicher Feature-DataFrame.
            features (list): Liste zu verwendender Spalten.
            categorical_cols (list): Liste der kategorialen Feature-Namen.

        Returns:
            pd.DataFrame: Modellfähiger Feature-Vektor.
    """
    df_features, _, _ = build_feature_frame(df, features, model_selection)
    return df_features




def model_and_scaler(model_selection,scaler_selection,cat_cols):
    """
        Gibt Modellobjekt und zugehörigen Skalierer zurück basierend auf Auswahl.

        Args:
            model_selection (str): Kürzel für Modelltyp ('lr', 'rf', 'xgb').
            scaler_selection (str): Skalierungsmethode ('Standard', 'MinMax', 'None').

        Returns:
            tuple: (Modellobjekt, Skalierungsobjekt oder 'passthrough')
    """
    return build_model_and_scaler(model_selection, scaler_selection, cat_cols)

def pca_explained_variance(X_train,model_selection,scaler_selection,categorical_dummy_cols,num_cols,cat_cols):
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

    return calculate_pca_explained_variance(
        X_train,
        model_selection,
        scaler_selection,
        categorical_dummy_cols,
        num_cols,
        cat_cols,
    )


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


def ml_pipe(pca_check,n_components,model_selection,model,scaler,categorical_dummy_cols,num_cols):
    """
        Baut eine ML-Pipeline mit Preprocessing, optional PCA und gewähltem Modell.

        Args:

            pca_check (bool): Ob PCA integriert werden soll.
            n_components (int): Anzahl PCA-Komponenten.
            model_selection (str): Modellkürzel.
            model: Modellobjekt.

        Returns:
            Pipeline: Sklearn-Pipeline für Training und Vorhersage.
    """
    return build_pipeline(
        pca_check,
        n_components,
        model_selection,
        model,
        scaler,
        categorical_dummy_cols,
        num_cols,
    )


def performance(y_test,y_pred):
    """
        Berechnet klassische Regressionsmetriken.

        Args:
            y_test (pd.Series): Wahre Zielwerte.
            y_pred (np.ndarray): Modellvorhersagen.

        Returns:
            tuple: (RMSE, R²-Score, MAE)
    """
    return calculate_regression_metrics(y_test, y_pred)


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







def feature_importance(pipe, model_selection,X_train):
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
    elif "preprocessor" in pipe.named_steps:
        last_transformer = pipe.named_steps["preprocessor"]
        feature_names = last_transformer.get_feature_names_out()
    else:
        feature_names = X_train.columns.tolist()

    if hasattr(model, "coef_"):
        importance = model.coef_
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        raise ValueError("Modelltyp nicht erkannt oder nicht gefittet.")

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    df_importance["Relative%"] = (df_importance["Importance"] / df_importance["Importance"].sum())*100

    return df_importance



def input_features(features):
    """
        Gibt die für die Vorhersage benötigten Feature-Widgets zurück, die für die Vorhersage benötigt werden.

        :param features: Alle features die für das Modell genutzt wurden
        :return: features die für das setzen der Widgets genutzt werden
    """
    return required_prediction_inputs(features)




def car_data(
    model_selection,
    categorical_dummy_cols,
    X_train,
    cat_cols,
    selected_features,
    hp=None,
    year=None,
    mileage=None,
    make=None,
    fuel=None,
    gear=None,
    model=None,
    offerType=None,
):
    """
        Erzeugt einen Feature-Vektor für ein einzelnes Fahrzeug basierend auf Benutzerangaben,
        passend zur Struktur des Trainingsdatensatzes (Dummy-kodierte Kategorien und numerische Merkmale).

        Args:
            categorical_dummy_cols (list): Liste aller Dummy-Features (One-Hot-kodierte Spaltennamen).
            X_train (pd.DataFrame): Trainingsdaten, um die Spaltenstruktur zu übernehmen.
            selected_features (list): Ursprünglich ausgewählte Features vor One-Hot-Encoding.
            hp (float, optional): Motorleistung in PS.
            year (int, optional): Erstzulassungsjahr.
            mileage (float, optional): Kilometerstand.
            make (str, optional): Fahrzeugmarke.
            fuel (str, optional): Kraftstofftyp.
            gear (str, optional): Getriebeart.
            model (str, optional): Modellbezeichnung.
            offerType (str, optional): Angebotstyp.

        Returns:
            pd.DataFrame: 1-zeiliger Feature-Vektor für Modellinput.
    """
    vehicle_inputs = {
        "hp": hp,
        "year": year,
        "mileage": mileage,
        "make": make,
        "fuel": fuel,
        "gear": gear,
        "model": model,
        "offerType": offerType,
    }

    return build_vehicle_input_frame(
        vehicle_inputs=vehicle_inputs,
        selected_features=selected_features,
        training_columns=X_train.columns.tolist(),
        categorical_columns=cat_cols,
        model_selection=model_selection,
    )
