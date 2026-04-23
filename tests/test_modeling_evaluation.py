import numpy as np
import pandas as pd

from autoscout24.data.io import load_modeling_dataset
from autoscout24.features.engineering import apply_feature_engineering
from autoscout24.modeling.evaluation import (
    SEGMENT_LABELS,
    EvaluationReport,
    build_evaluation_report,
    calculate_residual_diagnostics,
)


def test_build_evaluation_report_includes_requested_segments():
    base_df = load_modeling_dataset().head(50).copy()
    engineered_df = apply_feature_engineering(base_df)
    y_true = engineered_df["price"].iloc[:20]
    y_pred = y_true.to_numpy() * 0.95

    report = build_evaluation_report(
        y_true,
        y_pred,
        engineered_df,
    )

    assert set(report.segment_reports) == set(SEGMENT_LABELS)
    assert all(not frame.empty for frame in report.segment_reports.values())
    assert report.residual_diagnostics.delta_skew >= 0


def test_evaluation_report_roundtrip_preserves_segments():
    base_df = load_modeling_dataset().head(40).copy()
    engineered_df = apply_feature_engineering(base_df)
    y_true = engineered_df["price"].iloc[:10]
    y_pred = y_true.to_numpy() - 1_000

    report = build_evaluation_report(y_true, y_pred, engineered_df)
    restored = EvaluationReport.from_dict(report.to_dict())

    assert restored.y_true == report.y_true
    assert restored.segment_reports["offerType"].equals(report.segment_reports["offerType"])


def test_calculate_residual_diagnostics_returns_stable_metrics():
    diagnostics = calculate_residual_diagnostics(
        pd.Series([10_000, 20_000, 30_000, 40_000]),
        np.array([9_000, 21_000, 28_000, 39_000]),
    )

    assert isinstance(diagnostics.skew, float)
    assert diagnostics.delta_kurtosis >= 0
