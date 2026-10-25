import pandas as pd
from retail_forecast.baselines import (
    predict_previous_week,
    predict_trailing_mean,
)


def test_previous_week_predictions_preserve_missing_values_and_product_boundaries():
    weeks = pd.date_range("2009-12-07", periods=4, freq="W-MON")

    panel = pd.DataFrame(
        {
            "StockCode": ["A"] * 4 + ["B"] * 4,
            "week_start": weeks.tolist() * 2,
            "units_sold": pd.Series(
                [10, None, 30, 40, 100, 110, 120, 130],
                dtype="Float64",
            ),
        }
    )
    panel = panel.sample(frac=1, random_state=42).reset_index(drop=True)
    original = panel.copy(deep=True)

    result = predict_previous_week(panel)

    expected_predictions = pd.Series(
        [None, 10, None, 30, None, 100, 110, 120],
        name="prediction",
        dtype="Float64",
    )

    pd.testing.assert_series_equal(result["prediction"], expected_predictions)
    assert result["StockCode"].tolist() == ["A"] * 4 + ["B"] * 4
    pd.testing.assert_frame_equal(panel, original)


def test_trailing_mean_excludes_current_week_and_separates_products():
    weeks = pd.date_range("2009-12-07", periods=5, freq="W-MON")
    panel = pd.DataFrame(
        {
            "StockCode": ["A"] * 5 + ["B"] * 5,
            "week_start": weeks.tolist() * 2,
            "units_sold": pd.Series(
                [10, 20, 30, 40, 1000, 100, 200, 300, 400, 500],
                dtype="Float64",
            ),
        }
    )

    result = predict_trailing_mean(panel, window=4)

    expected = pd.Series(
        [None, None, None, None, 25, None, None, None, None, 250],
        name="prediction",
        dtype="Float64",
    )
    pd.testing.assert_series_equal(result["prediction"], expected)


def test_trailing_mean_requires_four_consecutive_known_weeks():
    panel = pd.DataFrame(
        {
            "StockCode": ["A"] * 8,
            "week_start": pd.date_range("2009-12-07", periods=8, freq="W-MON"),
            "units_sold": pd.Series(
                [10, 20, None, 40, 50, 60, 70, 80],
                dtype="Float64",
            ),
        }
    )

    result = predict_trailing_mean(panel, window=4)

    expected = pd.Series(
        [None] * 7 + [55],
        name="prediction",
        dtype="Float64",
    )
    pd.testing.assert_series_equal(result["prediction"], expected)
