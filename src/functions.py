from pathlib import Path

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st


@st.cache_data
def read_csv():
    """
        Lädt die  Autoscout24-Datendatei, entfernt Duplikate und NaNs.

        Returns:
            pd.DataFrame: Bereinigter Datensatz ohne Duplikate oder fehlende Werte.
    """
    script_path = Path(__file__).resolve()
    src_dir = script_path.parent
    data_file_path = src_dir / "autoscout24.csv"


    df = pd.read_csv(data_file_path)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    return df

@st.cache_data
def image_path():
    """
        Liefert den Pfad zur 'cardealer.png' Bilddatei im aktuellen Projektverzeichnis.

        Returns:
            pathlib.Path: Pfad zur Bilddatei.
    """
    script_path = Path(__file__).resolve()
    src_dir = script_path.parent
    image_file_path = src_dir / "cardealer.png"

    return image_file_path


@st.cache_data
def read_csv_with_nan_duplicates():
    """
        Lädt die Autoscout24-Datendatei inklusive Duplikaten und fehlender Werte.

        Returns:
            pd.DataFrame: Rohdatensatz mit NaN und Duplikaten.
    """


    script_path = Path(__file__).resolve()
    src_dir = script_path.parent  # == src/
    data_file_path = src_dir / "autoscout24.csv"
    df_with_nan = pd.read_csv(data_file_path)
   # df_with_nan.drop_duplicates(inplace=True)
    return df_with_nan

def show_nan(null_counts):
    """
        Erstellt eine Balkengrafik zur Visualisierung von NaN-Werten je Feature.

        Args:
            null_counts (pd.Series): Anzahl fehlender Werte pro Spalte.

        Returns:
            plotly.graph_objects.Figure: Balkendiagramm mit NaN-Verteilung.
    """

    fig = px.bar(null_counts,color_discrete_sequence=['#ff9249'])
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Anzahl NaN")
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False))
    return fig




def get_outliers(df_merged, df_filtered_iqr):
    """
        Berechnet pro Baujahr die Anzahl der entfernten Ausreißer.

        Args:
            df_merged (pd.DataFrame): Datensatz mit IQR-Grenzen.
            df_filtered_iqr (pd.DataFrame): Datensatz nach Entfernung der Ausreißer.

        Returns:
            pd.DataFrame: Übersicht der Ausreißeranzahl pro Jahr.
    """

    outliers = df_merged.groupby("year").size() - df_filtered_iqr.groupby("year").size()
    outliers_df = outliers.reset_index()
    outliers_df.columns = ["year", "outliers"]
    outliers_df = outliers_df.sort_values(by="outliers", ascending=False)

   # outliers_df["year"] = outliers_df["year"].astype(str)
    outliers_df["year"] = pd.Categorical(outliers_df["year"], categories=outliers_df["year"].tolist(), ordered=True)

    return outliers_df


def df_outliers_removed(df,factor=1.5):
    """
        Entfernt Ausreißer anhand eines IQR-Faktors für 'price' und 'mileage' je Baujahr.

        Args:
            df (pd.DataFrame): Ursprungsdatensatz mit Fahrzeugdaten.
            factor (float): Multiplikator für die IQR-Grenzen (Default: 1.5).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - df_merged: Datensatz inkl. IQR-Grenzen
                - df_filtered_iqr: Datensatz ohne Ausreißer
                - iqr_bounds: Grenzwerte für Preis und Laufleistung je Baujahr
    """
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

    hp_iqr_bounds = df_merged["hp"].quantile([0.25, 0.75])
    df_merged["hp_u_limit"] = hp_iqr_bounds[0.75]+(hp_iqr_bounds[0.75]-hp_iqr_bounds[0.25])*factor

    df_filtered_iqr = df_merged[
        (df_merged['price'] <= df_merged['price_u_limit']) & (df_merged['mileage'] <= df_merged['mileage_u_limit']) & (df_merged['hp'] <= df_merged['hp_u_limit'])]

    return df_merged,df_filtered_iqr,iqr_bounds



def plot_outliers(outliers_df,iqr_bounds):
    """
        Visualisiert Ausreißer pro Baujahr sowie IQR-basierte Preis- und Laufleistungsgrenzen.

        Args:
            outliers_df (pd.DataFrame): Anzahl der Ausreißer je Jahr.
            iqr_bounds (pd.DataFrame): IQR-Grenzen für Preis und Mileage.

        Returns:
            plotly.graph_objects.Figure: Kombinierte Balken- und Liniendiagramme.
    """

    vivid_colors = px.colors.qualitative.Vivid[:3]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=outliers_df["year"], y=outliers_df["outliers"], name="Anzahl Ausreißer",marker_color=vivid_colors[0]))
    fig.add_trace(go.Scatter(
        x=iqr_bounds["year"],
        y=iqr_bounds["price_u_limit"],
        mode="lines",
        name="Oberes Preislimit",
        yaxis="y2",line=dict(color=vivid_colors[1], width=2)
    ))


    fig.add_trace(go.Scatter(
        x=iqr_bounds["year"],
        y=iqr_bounds["mileage_u_limit"],
        mode="lines",
        name="Oberes Mileage-Limit",
        yaxis="y2", line=dict(color=vivid_colors[2], width=2)
    ))


    fig.update_layout(
        yaxis=dict(
            title="Anzahl Ausreißer",
            showgrid=False
        ),
        yaxis2=dict(
            title="Limit (Preis/Mileage)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis=dict(title=""),
        legend=dict(x=0.75, y=1.2, xanchor='right', yanchor='top')
    )
    fig.update_layout(
        hovermode="x unified"
    )
    fig.update_layout(
        width=900,
        height=400)
    fig.update_layout(
        yaxis=dict(range=[0, 600]),
       # yaxis2=dict(range=[0, 80000])
    )
    return fig






def layout():
    settings = {
        "hovermode": "x unified",
        "uniformtext_minsize": 16,
        # Font-Größen
        "yaxis_title_font_size": 16,
        "legend_font_size": 16,
        "xaxis_tickfont_size": 16,
        "yaxis_tickfont_size": 16,
        # Grid entfernen
        "xaxis": {"showgrid": False},  # Hier muss xaxis auch ein dict sein
        "yaxis": {"showgrid": False}  # Hier muss yaxis auch ein dict sein
    }
    return settings

"""
def plot_sales(df_sales):
    fig = px.area(df_sales, x="year", y="size",color="offerType", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Anzahl verkaufter Autos")
    fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>" +
                                    "%{y:,.0f}<extra></extra>")
    fig.update_layout(**layout())
    #fig.update_yaxes(range=[3500, 4500])
    return fig
"""

def plot_make_pie(df,cat,years):
    """
        Erstellt für jedes Jahr ein Kreisdiagramm der Verkaufsanteile je Kategorie (z.B. Marke).

        Args:
            df (pd.DataFrame): Datensatz mit Verkaufsinformationen.
            cat (str): Kategorie zum Gruppieren (z. B. 'make').
            years (List[int]): Liste der Jahrgänge zur Analyse.

        Returns:
            None: Diagramme werden über Streamlit direkt angezeigt.
    """
    cols = st.columns(len(years))
    for i,year in enumerate(years):
        sales = df[df["year"].isin([year])].groupby([cat, "year"]).size().reset_index(name="count")
        fig = px.pie(sales, names=cat,values="count", color_discrete_sequence=px.colors.qualitative.Vivid, hole=0.2)
        fig.update_layout(showlegend=False)

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            texttemplate='%{label}<br>%{percent:.1%}',
            # --- THIS IS THE KEY FOR FONT SIZE IN SLICES ---
            textfont_size=16  # Adjust this value to your desired font size
        )

        fig.update_traces(
            # hovertemplate zum Anpassen des Hover-Textes
            hovertemplate="<b>Marke: %{label}</b><br>" +
                          "Anteil: %{percent:.1%}<extra></extra>")
        fig.update_xaxes(title_text="")
        fig.update_annotations(text="")
        with cols[i]:
            sales = df[df["year"]==year]["make"].size
            st.markdown(f"""**Verkäufe Erstzulassung {year}    
                       {sales}**""")
            st.plotly_chart(fig, key=f"pie_{cat}_{year}")


def plot_make_pie_sales_volume(df,cat,years):
    """
        Visualisiert für jedes Jahr den Umsatzanteil je Kategorie (z.B Marke) als Kreisdiagramm.

        Args:
            df (pd.DataFrame): Datensatz mit Preisinformationen.
            cat (str): Kategorie zum Gruppieren (z. B. 'make').
            years (List[int]): Liste der Jahrgänge zur Analyse.

        Returns:
            None: Umsatzdiagramme werden über Streamlit direkt angezeigt.
    """

    cols = st.columns(len(years))
    for i,year in enumerate(years):
        sales_filter = df[df["year"].isin([year])].groupby([cat, "year"])["price"].sum().reset_index(
            name="sales_volume")
        fig = px.pie(sales_filter, names=cat,values="sales_volume", color_discrete_sequence=px.colors.qualitative.Vivid,hole=0.2)
        fig.update_layout(showlegend=False)

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            texttemplate='%{label}<br>%{percent:.1%}',
            # --- THIS IS THE KEY FOR FONT SIZE IN SLICES ---
            textfont_size=16  # Adjust this value to your desired font size
        )
        fig.for_each_yaxis(lambda axis: axis.update(title_text="", showticklabels=False))

        fig.update_traces(
            # hovertemplate zum Anpassen des Hover-Textes
            hovertemplate="<b>Marke: %{label}</b><br>" +
                          "Anteil: %{percent:.1%}<extra></extra>")
        fig.update_xaxes(title_text="")
        fig.update_annotations(text="")
        with cols[i]:
            sales_volume=df[df["year"]==year]["price"].sum()/1000000
            st.markdown(f"""**Preis Gesamt Erstzulassung {year}    
                       {sales_volume:.2f} Mio.**""")
            st.plotly_chart(fig, key=f"pie_sales_volume_{cat}_{year}")









"""
def plot_offer_cat(df_offer_cat,cat,options):

    fig = px.area(df_offer_cat[df_offer_cat[cat].isin(options)], x="year", y="size", color="offerType",
                  color_discrete_sequence=px.colors.qualitative.Pastel, facet_col=cat,
                  facet_col_wrap=4)  # ,groupnorm="percent")
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Anzahl verkaufter Autos")
    fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>" +
                                    "%{y:,.0f}<extra></extra>")


    fig.update_xaxes(title_text="", matches=None, showgrid=False,
                     tickfont=dict(size=14), title=dict(font=dict(size=14)))
    # Alle Y-Achsen beschriften (Ticks und Zahlen), aber ohne Titel
    for axis in fig.layout:
        if axis.startswith("yaxis"):
            fig.layout[axis].showticklabels = True
            fig.layout[axis].tickfont = dict(size=14)
            fig.layout[axis].title.text = ""  # leeren Titel setzen
            fig.layout[axis].showgrid= False


    # Nur die erste Y-Achse bekommt einen Titel
    fig.layout.yaxis.title.text = "Anzahl verkaufter Autos"
    fig.update_layout(hovermode="x unified")

    return fig
"""

def top_10_sales(df,fuel,gear,year):
    """
        Berechnet die Top-10 meistverkauften Fahrzeugmarken je Kraftstofftyp und Getriebe (optional nach Jahr).

        Args:
            df (pd.DataFrame): Gesamtfahrzeug-Datensatz.
            fuel (list): Ausgewählte Kraftstofftypen.
            gear (list): Ausgewählte Getriebearten.
            year (list): Liste von Jahrgängen (z. B. ['2015', '2016'] oder ['alle']).

        Returns:
            pd.DataFrame: Top 10 Verkäufe je Gruppe (Make, Fuel, Gear[, Year]) mit Verkaufsanzahl.
    """
    df["year"] = df["year"].astype(str)
    if year[0]=="alle":
        df_filtered = df[(df["fuel"].isin(fuel)) & (df["gear"].isin(gear))]
        df_counted = (
            df_filtered
            .groupby(["make", "fuel", "gear"])
            .size()
            .reset_index(name="count"))
        top10 = (
            df_counted
            .groupby(["fuel", "gear"])
            .apply(lambda g: g.nlargest(10, "count"))
            .reset_index(drop=True)
        )

    else:
        df_filtered = df[(df["fuel"].isin(fuel)) & (df["gear"].isin(gear)) & (df["year"].isin(year))]
        df_counted = (
            df_filtered
            .groupby(["year", "make", "fuel", "gear"])
            .size()
            .reset_index(name="count"))
        top10 = (
            df_counted
            .groupby(["year", "fuel", "gear"])
            .apply(lambda g: g.nlargest(10, "count"))
            .reset_index(drop=True)
        )


    return top10

def plot_top_10(top10,year):
    """
        Visualisiert die Top-10 meistverkauften Marken als gruppierte Balkendiagramme,
        facettiert nach Kraftstoff und Getriebe. Optional pro Jahr oder übergreifend.

        Args:
            top10 (pd.DataFrame): Tabelle mit Top-Verkäufen.
            year (list): Liste mit Jahrgangsfilter (z. B. ['2015'] oder ['alle']).

        Returns:
            plotly.graph_objects.Figure: Interaktive Balkengrafik.
    """

    if year[0]!="alle":
        fig=px.bar(top10,x="year",y="count",color="make",
                   color_discrete_sequence=px.colors.qualitative.Vivid,
                   facet_col="fuel",facet_row="gear",
                   facet_row_spacing=0.1,
                   facet_col_spacing=0.05,barmode="group")
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Jahr: %{x}<br>" +
                          "Anzahl: %{y}<extra></extra>"
        )


    else:
        fig = px.bar(top10, x="make", y="count", color="make",
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     facet_col="fuel", facet_row="gear",
                     facet_row_spacing=0.1,
                     facet_col_spacing=0.05, barmode="group")
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Marke: %{x}<br>" +
                          "Anzahl: %{y}<extra></extra>"
        )

    fig.update_layout(showlegend=True)

    fig.update_xaxes(title_text="", matches=None, showgrid=False,
                         tickfont=dict(size=14), title=dict(font=dict(size=14)))
    for axis in fig.layout:
        if axis.startswith("yaxis"):
            fig.layout[axis].showticklabels = True
            fig.layout[axis].tickfont = dict(size=14)
            fig.layout[axis].title.text = ""  # leeren Titel setzen
            fig.layout[axis].showgrid= False
    fig.layout.yaxis.title.text = "Anzahl verkaufter Autos"

    return fig




def plot_rel_price_box(data_frame,color_value,mileage=False,price_mileage=False):
    """
        Visualisiert Preise oder Laufleistungen als Boxplot oder Preis-Mileage-Korrelation als Streudiagramm.

        Args:
            data_frame (pd.DataFrame): Datensatz mit Fahrzeugangeboten.
            color_value (str): Spalte für Farbgruppen (z. B. 'make').
            mileage (bool): Wenn True → Boxplot für Mileage.
            price_mileage (bool): Wenn True → Scatterplot für Preis vs. Mileage.
            (Standard): Boxplot für Preis nach Jahr.

        Returns:
            plotly.graph_objects.Figure: Interaktive Visualisierung.
    """
    if mileage:
        fig = px.box(data_frame=data_frame, x="year", y="mileage", color=color_value, facet_col="fuel",facet_row="gear",color_discrete_sequence=px.colors.qualitative.Vivid,
                     facet_col_spacing=0.2, width=100, height=600)
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Jahr: %{x}<br>" +
                          "Mileage: %{y}<extra></extra>"
        )

        fig.update_xaxes(title_text="", matches=None, showgrid=False,
                         tickfont=dict(size=14), title=dict(font=dict(size=14)))
        for axis in fig.layout:
            if axis.startswith("yaxis"):
                fig.layout[axis].showticklabels = True
                fig.layout[axis].tickfont = dict(size=14)
                fig.layout[axis].title.text = ""  # leeren Titel setzen
                fig.layout[axis].showgrid = False
        fig.layout.yaxis.title.text = "Mileage"

    elif price_mileage:
        fig=px.scatter(data_frame,x="mileage",y="price",color=color_value,facet_col="fuel", facet_col_spacing=0.2, width=100, height=600,facet_row="gear",
                       color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Mileage: %{x}<br>" +
                          "Preis: %{y}<extra></extra>"
        )

        fig.update_xaxes(title_text="", matches=None, showgrid=False,
                         tickfont=dict(size=14), title=dict(font=dict(size=14)))
        for axis in fig.layout:
            if axis.startswith("yaxis"):
                fig.layout[axis].showticklabels = True
                fig.layout[axis].tickfont = dict(size=14)
                fig.layout[axis].title.text = ""  # leeren Titel setzen
                fig.layout[axis].showgrid = False
        fig.layout.yaxis.title.text = "Preis"


    else:
        fig = px.box(data_frame=data_frame, x="year", y="price", color=color_value, facet_col="fuel",facet_row="gear",
                     facet_col_spacing=0.2, width=100, height=600,
                     color_discrete_sequence=px.colors.qualitative.Vivid)

        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Jahr: %{x}<br>" +
                          "Preis: %{y}<extra></extra>"
        )

        fig.update_xaxes(title_text="", matches=None, showgrid=False,
                         tickfont=dict(size=14), title=dict(font=dict(size=14)))
        for axis in fig.layout:
            if axis.startswith("yaxis"):
                fig.layout[axis].showticklabels = True
                fig.layout[axis].tickfont = dict(size=14)
                fig.layout[axis].title.text = ""  # leeren Titel setzen
                fig.layout[axis].showgrid = False
        fig.layout.yaxis.title.text = "Preis"

    return fig






