import pandas as pd

from autoscout24.exploration.charts import build_category_count_chart


def test_build_category_count_chart_labels_missing_values():
    df = pd.DataFrame({"gear": ["Manual", None, "Automatic", None]})

    fig = build_category_count_chart(df, "gear")

    assert len(fig.data) == 1
    counts = dict(zip(fig.data[0].y, fig.data[0].x, strict=False))
    assert counts == {"Unbekannt": 2, "Automatic": 1, "Manual": 1}
