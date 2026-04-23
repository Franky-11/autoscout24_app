import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_null_counts_chart(null_counts: pd.Series) -> go.Figure:
    fig = px.bar(null_counts, color_discrete_sequence=["#ff9249"])
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Anzahl NaN")
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
    )
    return fig


def build_outlier_chart(outliers_df: pd.DataFrame, bounds_df: pd.DataFrame) -> go.Figure:
    colors = px.colors.qualitative.Vivid[:3]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=outliers_df["year"],
            y=outliers_df["outliers"],
            name="Anzahl Ausreißer",
            marker_color=colors[0],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bounds_df["year"],
            y=bounds_df["price_u_limit"],
            mode="lines",
            name="Oberes Preislimit",
            yaxis="y2",
            line={"color": colors[1], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bounds_df["year"],
            y=bounds_df["mileage_u_limit"],
            mode="lines",
            name="Oberes Mileage-Limit",
            yaxis="y2",
            line={"color": colors[2], "width": 2},
        )
    )
    fig.update_layout(
        yaxis={"title": "Anzahl Ausreißer", "showgrid": False, "range": [0, 600]},
        yaxis2={
            "title": "Limit (Preis/Mileage)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        xaxis={"title": ""},
        legend={"x": 0.75, "y": 1.2, "xanchor": "right", "yanchor": "top"},
        hovermode="x unified",
        width=900,
        height=400,
    )
    return fig


def build_category_share_chart(
    df: pd.DataFrame,
    category: str,
    value_column: str,
    hover_label: str,
) -> go.Figure:
    fig = px.pie(
        df,
        names=category,
        values=value_column,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        hole=0.2,
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        texttemplate="%{label}<br>%{percent:.1%}",
        textfont_size=16,
        hovertemplate=(
            f"<b>{hover_label}: %{{label}}</b><br>"
            "Anteil: %{percent:.1%}<extra></extra>"
        ),
    )
    fig.update_xaxes(title_text="")
    fig.update_annotations(text="")
    return fig


def build_top_sales_chart(top_sales_df: pd.DataFrame, include_year_dimension: bool) -> go.Figure:
    if include_year_dimension:
        fig = px.bar(
            top_sales_df,
            x="year",
            y="count",
            color="make",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            facet_col="fuel",
            facet_row="gear",
            facet_row_spacing=0.1,
            facet_col_spacing=0.05,
            barmode="group",
        )
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>Jahr: %{x}<br>Anzahl: %{y}<extra></extra>"
        )
    else:
        fig = px.bar(
            top_sales_df,
            x="make",
            y="count",
            color="make",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            facet_col="fuel",
            facet_row="gear",
            facet_row_spacing=0.1,
            facet_col_spacing=0.05,
            barmode="group",
        )
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>Marke: %{x}<br>Anzahl: %{y}<extra></extra>"
        )

    fig.update_layout(showlegend=True)
    fig.update_xaxes(
        title_text="",
        matches=None,
        showgrid=False,
        tickfont={"size": 14},
        title={"font": {"size": 14}},
    )
    for axis in fig.layout:
        if axis.startswith("yaxis"):
            fig.layout[axis].showticklabels = True
            fig.layout[axis].tickfont = {"size": 14}
            fig.layout[axis].title.text = ""
            fig.layout[axis].showgrid = False
    fig.layout.yaxis.title.text = "Anzahl verkaufter Autos"
    return fig


def build_relation_chart(
    data_frame: pd.DataFrame,
    color_value: str | None,
    *,
    mode: str,
) -> go.Figure:
    common_kwargs = {
        "data_frame": data_frame,
        "color": color_value,
        "facet_col": "fuel",
        "facet_row": "gear",
        "facet_col_spacing": 0.2,
        "width": 100,
        "height": 600,
    }
    colors = px.colors.qualitative.Vivid

    if mode == "mileage_year":
        fig = px.box(
            x="year",
            y="mileage",
            color_discrete_sequence=colors,
            **common_kwargs,
        )
        hover_template = "<b>%{fullData.name}</b><br>Jahr: %{x}<br>Mileage: %{y}<extra></extra>"
        y_title = "Mileage"
    elif mode == "price_mileage":
        fig = px.scatter(
            x="mileage",
            y="price",
            color_discrete_sequence=colors,
            **common_kwargs,
        )
        hover_template = "<b>%{fullData.name}</b><br>Mileage: %{x}<br>Preis: %{y}<extra></extra>"
        y_title = "Preis"
    else:
        fig = px.box(
            x="year",
            y="price",
            color_discrete_sequence=colors,
            **common_kwargs,
        )
        hover_template = "<b>%{fullData.name}</b><br>Jahr: %{x}<br>Preis: %{y}<extra></extra>"
        y_title = "Preis"

    fig.update_traces(hovertemplate=hover_template)
    fig.update_xaxes(
        title_text="",
        matches=None,
        showgrid=False,
        tickfont={"size": 14},
        title={"font": {"size": 14}},
    )
    for axis in fig.layout:
        if axis.startswith("yaxis"):
            fig.layout[axis].showticklabels = True
            fig.layout[axis].tickfont = {"size": 14}
            fig.layout[axis].title.text = ""
            fig.layout[axis].showgrid = False
    fig.layout.yaxis.title.text = y_title
    return fig
