from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import kurtosis, probplot, skew

PRICE_BAND_BINS = [0, 10_000, 20_000, 30_000, 40_000, 50_000, 75_000, np.inf]
PRICE_BAND_LABELS = ["0-10k", "10k-20k", "20k-30k", "30k-40k", "40k-50k", "50k-75k", "75k+"]
CAR_AGE_BAND_BINS = [-0.1, 1, 4, 7, 10, np.inf]
CAR_AGE_BAND_LABELS = ["0-1", "2-4", "5-7", "8-10", "11+"]
SEGMENT_LABELS = {
    "price_band": "Preisband",
    "make": "Marke",
    "offerType": "Angebotstyp",
    "car_age_band": "Fahrzeugalter",
}


@dataclass(frozen=True)
class ResidualDiagnostics:
    skew: float
    kurtosis: float
    delta_skew: float
    delta_kurtosis: float


@dataclass(frozen=True)
class EvaluationReport:
    y_true: list[float]
    y_pred: list[float]
    residual_diagnostics: ResidualDiagnostics
    segment_reports: dict[str, pd.DataFrame]

    def to_dict(self) -> dict[str, object]:
        return {
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "residual_diagnostics": asdict(self.residual_diagnostics),
            "segment_reports": {
                key: frame.to_dict(orient="records")
                for key, frame in self.segment_reports.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EvaluationReport":
        diagnostics = ResidualDiagnostics(**data["residual_diagnostics"])
        segment_reports = {
            key: pd.DataFrame(records)
            for key, records in data.get("segment_reports", {}).items()
        }
        return cls(
            y_true=list(data.get("y_true", [])),
            y_pred=list(data.get("y_pred", [])),
            residual_diagnostics=diagnostics,
            segment_reports=segment_reports,
        )


def build_evaluation_report(
    y_true: pd.Series,
    y_pred: np.ndarray | pd.Series,
    reference_df: pd.DataFrame,
) -> EvaluationReport:
    evaluation_df = _build_evaluation_frame(y_true, y_pred, reference_df)
    segment_reports = {
        "price_band": _summarize_segments(evaluation_df, "price_band"),
        "make": _summarize_segments(evaluation_df, "make"),
        "offerType": _summarize_segments(evaluation_df, "offerType"),
        "car_age_band": _summarize_segments(evaluation_df, "car_age_band"),
    }
    diagnostics = calculate_residual_diagnostics(
        evaluation_df["actual_price"],
        evaluation_df["predicted_price"],
    )
    return EvaluationReport(
        y_true=evaluation_df["actual_price"].astype(float).tolist(),
        y_pred=evaluation_df["predicted_price"].astype(float).tolist(),
        residual_diagnostics=diagnostics,
        segment_reports=segment_reports,
    )


def calculate_residual_diagnostics(
    y_true: pd.Series | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> ResidualDiagnostics:
    residuals = pd.Series(y_true, dtype=float) - pd.Series(y_pred, dtype=float)
    if residuals.nunique(dropna=True) <= 1:
        return ResidualDiagnostics(
            skew=0.0,
            kurtosis=3.0,
            delta_skew=0.0,
            delta_kurtosis=0.0,
        )
    res_skew = float(skew(residuals))
    res_kurt = float(kurtosis(residuals, fisher=False))
    return ResidualDiagnostics(
        skew=res_skew,
        kurtosis=res_kurt,
        delta_skew=abs(res_skew),
        delta_kurtosis=abs(res_kurt - 3),
    )


def build_prediction_scatter(
    y_true: pd.Series | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> go.Figure:
    true_values = pd.Series(y_true, dtype=float)
    predicted_values = pd.Series(y_pred, dtype=float)
    fig = go.Figure()
    plot_color = px.colors.qualitative.Vivid[0]
    fig.add_trace(
        go.Scatter(
            x=true_values,
            y=predicted_values,
            mode="markers",
            marker={"color": plot_color, "opacity": 0.7},
            name="Vorhersage vs. Ist-Wert",
        )
    )

    max_val = max(true_values.max(), predicted_values.max())
    min_val = min(true_values.min(), predicted_values.min())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"color": "grey", "dash": "dash"},
            name="Perfekte Vorhersage (Y=X)",
        )
    )
    fig.update_layout(
        xaxis_title="Wahrer Preis",
        yaxis_title="Vorhergesagter Preis",
        height=555,
        hovermode="closest",
    )
    return fig


def build_pca_explained_variance_chart(explained_variance: np.ndarray | pd.Series) -> go.Figure:
    df_var = pd.DataFrame(
        {
            "Anzahl Komponenten": np.arange(1, len(explained_variance) + 1),
            "Kumulierte erklärte Varianz": explained_variance,
        }
    )
    plot_color = px.colors.qualitative.Vivid[0]
    fig = px.line(
        df_var,
        x="Anzahl Komponenten",
        y="Kumulierte erklärte Varianz",
        labels={
            "Kumulierte erklärte Varianz": "Kumulierte erklärte Varianz",
            "Anzahl Komponenten": "Anzahl der Komponenten",
        },
    )
    fig.update_traces(
        line_color=plot_color,
        marker_color=plot_color,
        mode="lines",
        line_width=3,
        marker_size=8,
    )
    fig.update_layout(
        showlegend=False,
        xaxis={"title_font": {"size": 16}, "tickfont": {"size": 14}, "showgrid": True},
        yaxis={"title_font": {"size": 16}, "tickfont": {"size": 14}, "showgrid": True},
        hovermode="x unified",
    )
    return fig


def build_qq_plot(
    y_true: pd.Series | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> go.Figure:
    true_values = pd.Series(y_true, dtype=float)
    predicted_values = pd.Series(y_pred, dtype=float)
    residuals = true_values - predicted_values
    residual_std = residuals.std()
    if residual_std == 0 or pd.isna(residual_std):
        residuals_standard = pd.Series(np.zeros(len(residuals)))
    else:
        residuals_standard = (residuals - residuals.mean()) / residual_std
    qq_x, qq_y = probplot(residuals_standard, dist="norm", fit=False)
    plot_color = px.colors.qualitative.Vivid[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qq_x,
            y=qq_y,
            mode="markers",
            name="Beobachtete Residuen-Quantile",
            marker={"color": plot_color, "opacity": 0.7},
            hovertemplate="Theoretisch: %{x:.2f}<br>Beobachtet: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min(qq_x), max(qq_x)],
            y=[min(qq_x), max(qq_x)],
            mode="lines",
            name="Normalverteilungs-Referenz",
            line={"color": "grey", "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis={
            "title": "Theoretische Quantile (Normalverteilung)",
            "showgrid": False,
        },
        yaxis={"title": "Beobachtete Quantile (Residuen)", "showgrid": False},
    )
    return fig


def build_segment_error_chart(
    segment_df: pd.DataFrame,
    *,
    title: str,
) -> go.Figure:
    fig = px.bar(
        segment_df,
        x="Segment",
        y="Bias",
        color="Bias",
        color_continuous_scale="RdBu",
        text="Bias",
        custom_data=["Anzahl", "MAE", "RMSE", "Median AE"],
        title=title,
    )
    fig.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        hovertemplate=(
            "Segment: %{x}<br>"
            "Bias: %{y:.0f} EUR<br>"
            "Anzahl: %{customdata[0]}<br>"
            "MAE: %{customdata[1]:.0f} EUR<br>"
            "RMSE: %{customdata[2]:.0f} EUR<br>"
            "Median AE: %{customdata[3]:.0f} EUR<extra></extra>"
        ),
    )
    max_abs_bias = max(abs(segment_df["Bias"].min()), abs(segment_df["Bias"].max()))
    fig.update_coloraxes(cmin=-max_abs_bias, cmax=max_abs_bias)
    fig.update_layout(yaxis_title="Bias", xaxis_title="", height=450)
    return fig


def calculate_feature_importance(
    pipeline: object,
    model_selection: str,
    x_train: pd.DataFrame,
) -> pd.DataFrame:
    model = pipeline.named_steps[model_selection]

    if "pca" in pipeline.named_steps:
        n_components = pipeline.named_steps["pca"].n_components_
        feature_names = [f"PC{i + 1}" for i in range(n_components)]
    elif "preprocessor" in pipeline.named_steps:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    else:
        feature_names = x_train.columns.tolist()

    if hasattr(model, "coef_"):
        importance = model.coef_
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        raise ValueError("Modelltyp nicht erkannt oder nicht gefittet.")

    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values(
        by="Importance",
        ascending=False,
    )
    total_importance = importance_df["Importance"].sum()
    if total_importance == 0:
        importance_df["Relative%"] = 0.0
    else:
        importance_df["Relative%"] = importance_df["Importance"] / total_importance * 100
    return importance_df


def _build_evaluation_frame(
    y_true: pd.Series,
    y_pred: np.ndarray | pd.Series,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    aligned_reference = reference_df.loc[y_true.index].copy()
    evaluation_df = aligned_reference.assign(
        actual_price=y_true.astype(float),
        predicted_price=pd.Series(y_pred, index=y_true.index, dtype=float),
    )
    evaluation_df["residual"] = evaluation_df["actual_price"] - evaluation_df["predicted_price"]
    evaluation_df["abs_error"] = evaluation_df["residual"].abs()
    evaluation_df["price_band"] = pd.cut(
        evaluation_df["actual_price"],
        bins=PRICE_BAND_BINS,
        labels=PRICE_BAND_LABELS,
        include_lowest=True,
        ordered=True,
    )
    evaluation_df["car_age_band"] = pd.cut(
        evaluation_df["car_age"],
        bins=CAR_AGE_BAND_BINS,
        labels=CAR_AGE_BAND_LABELS,
        include_lowest=True,
        ordered=True,
    )
    return evaluation_df


def _summarize_segments(evaluation_df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = evaluation_df.dropna(subset=[column]).groupby(column, observed=False)
    summary = grouped.agg(
        Anzahl=("actual_price", "size"),
        MAE=("abs_error", "mean"),
        **{"Median AE": ("abs_error", "median"), "Bias": ("residual", "mean")},
    )
    summary["RMSE"] = grouped["residual"].apply(lambda values: float(np.sqrt(np.mean(values**2))))
    summary = summary.reset_index().rename(columns={column: "Segment"})

    summary = summary[summary["Anzahl"] > 0].copy()
    summary["Segment"] = summary["Segment"].astype(str)

    if isinstance(evaluation_df[column].dtype, pd.CategoricalDtype):
        summary = summary.sort_values("Segment")
    else:
        summary = summary.sort_values(["Anzahl", "Segment"], ascending=[False, True])

    return summary.reset_index(drop=True)
