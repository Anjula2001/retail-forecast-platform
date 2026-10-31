import pandas as pd
from retail_forecast.features import add_lag_features


def test_lag_features_use_previous_week_and_keep_products_separate():
    weeks = pd.date_range("2009-12-07", periods=4, freq="W-MON")

    panel = pd.DataFrame(
        {
            "StockCode": ["A"] * 4 + ["B"] * 4,
            "week_start": weeks.tolist() * 2,
            "units_sold": pd.Series(
                [10, None, 30, 1000, 100, 110, 120, 130],
                dtype="Float64",
            ),
        }
    )
    panel = panel.sample(frac=1, random_state=42).reset_index(drop=True)
    original = panel.copy(deep=True)

    result = add_lag_features(panel)

    expected = pd.Series(
        [None, 10, None, 30, None, 100, 110, 120],
        name="lag_1",
        dtype="Float64",
    )

    pd.testing.assert_series_equal(result["lag_1"], expected)
    assert result["StockCode"].tolist() == ["A"] * 4 + ["B"] * 4
    pd.testing.assert_frame_equal(panel, original)

    expected_lag_2 = pd.Series(
        [None, None, 10, None, None, None, 100, 110],
        name="lag_2",
        dtype="Float64",
    )
    expected_lag_4 = pd.Series(
        [None] * 8,
        name="lag_4",
        dtype="Float64",
    )

    pd.testing.assert_series_equal(result["lag_2"], expected_lag_2)
    pd.testing.assert_series_equal(result["lag_4"], expected_lag_4)


def test_lag_four_uses_sales_exactly_four_weeks_earlier():
    panel = pd.DataFrame(
        {
            "StockCode": ["A"] * 5,
            "week_start": pd.date_range("2009-12-07", periods=5, freq="W-MON"),
            "units_sold": pd.Series(
                [10, 20, 30, 40, 1000],
                dtype="Float64",
            ),
        }
    )

    result = add_lag_features(panel)

    assert result.loc[4, "lag_4"] == 10


def test_rolling_mean_uses_four_previous_known_weeks_per_product():
    weeks = pd.date_range("2009-12-07", periods=6, freq="W-MON")

    panel = pd.DataFrame(
        {
            "StockCode": ["A"] * 6 + ["B"] * 6,
            "week_start": weeks.tolist() * 2,
            "units_sold": pd.Series(
                [10, 20, 30, 40, 1000, 60, None, 20, 30, 40, 50, 60],
                dtype="Float64",
            ),
        }
    )

    result = add_lag_features(panel)

    expected = pd.Series(
        [
            None,
            None,
            None,
            None,
            25,
            272.5,
            None,
            None,
            None,
            None,
            None,
            35,
        ],
        name="rolling_mean_4",
        dtype="Float64",
    )

    pd.testing.assert_series_equal(result["rolling_mean_4"], expected)
