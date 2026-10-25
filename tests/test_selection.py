import pandas as pd
from retail_forecast.selection import (
    select_training_products,
    summarize_training_products,
)


def test_product_summary_uses_only_training_period():
    weekly = pd.DataFrame(
        {
            "StockCode": ["001A", "001A", "001A", "001A", "002B"],
            "week_start": pd.to_datetime(
                [
                    "2009-11-30",
                    "2009-12-07",
                    "2011-05-30",
                    "2011-06-06",
                    "2011-09-05",
                ]
            ),
            "units_sold": [100, 2, 3, 1000, 2000],
        }
    )
    original = weekly.copy(deep=True)

    result = summarize_training_products(
        weekly,
        train_start=pd.Timestamp("2009-12-07"),
        train_end_exclusive=pd.Timestamp("2011-06-06"),
    )

    assert result["StockCode"].tolist() == ["001A"]
    assert result["sales_weeks"].tolist() == [2]
    assert result["total_units"].tolist() == [5]
    assert result["first_sales_week"].tolist() == [pd.Timestamp("2009-12-07")]
    assert result["last_sales_week"].tolist() == [pd.Timestamp("2011-05-30")]
    pd.testing.assert_frame_equal(weekly, original)


def test_product_selection_applies_eligibility_before_ranking():
    weekly = pd.DataFrame(
        {
            "StockCode": ["A", "B", "B", "C", "C", "Z"],
            "week_start": pd.to_datetime(
                [
                    "2009-12-07",
                    "2009-12-07",
                    "2009-12-14",
                    "2009-12-07",
                    "2009-12-14",
                    "2011-06-06",
                ]
            ),
            "units_sold": [1000, 2, 3, 1, 2, 99999],
        }
    )

    selected = select_training_products(
        weekly,
        train_start=pd.Timestamp("2009-12-07"),
        train_end_exclusive=pd.Timestamp("2011-06-06"),
        n_products=1,
        min_sales_weeks=2,
    )

    assert selected["StockCode"].tolist() == ["B"]
    assert selected["total_units"].tolist() == [5]
