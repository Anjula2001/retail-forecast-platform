import pandas as pd
from retail_forecast.aggregation import (
    aggregate_observed_weekly_sales,
    build_weekly_panel,
    select_full_calendar_weeks,
)


def test_weekly_sales_separates_products_and_sunday_monday_boundary():
    sales = pd.DataFrame(
        {
            "StockCode": ["001A", "001A", "001A", "002B", "002B"],
            "InvoiceDate": pd.to_datetime(
                [
                    "2010-12-05",
                    "2010-12-06",
                    "2010-12-08",
                    "2010-12-05",
                    "2010-12-06",
                ]
            ),
            "Quantity": [2, 3, 4, 5, 6],
        }
    )
    original = sales.copy(deep=True)

    result = aggregate_observed_weekly_sales(sales)

    expected = pd.DataFrame(
        {
            "StockCode": ["001A", "001A", "002B", "002B"],
            "week_start": pd.to_datetime(["2010-11-29", "2010-12-06", "2010-11-29", "2010-12-06"]),
            "units_sold": [2, 7, 5, 6],
        }
    )

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(sales, original)
    assert result["units_sold"].sum() == sales["Quantity"].sum()


def test_select_full_calendar_weeks_excludes_partial_and_outside_weeks():
    weekly = pd.DataFrame(
        {
            "StockCode": ["001A"] * 4,
            "week_start": pd.to_datetime(["2009-11-30", "2009-12-07", "2009-12-14", "2009-12-21"]),
            "units_sold": [10, 20, 30, 40],
        }
    )
    original = weekly.copy(deep=True)

    result = select_full_calendar_weeks(
        weekly,
        coverage_start=pd.Timestamp("2009-12-01"),
        coverage_end_exclusive=pd.Timestamp("2009-12-21"),
    )

    assert result["week_start"].tolist() == [
        pd.Timestamp("2009-12-07"),
        pd.Timestamp("2009-12-14"),
    ]
    assert result["units_sold"].tolist() == [20, 30]
    assert result.index.tolist() == [0, 1]
    pd.testing.assert_frame_equal(weekly, original)


def test_weekly_panel_preserves_missing_weeks_and_selected_products():
    weekly = pd.DataFrame(
        {
            "StockCode": ["A", "A", "B", "B", "C"],
            "week_start": pd.to_datetime(
                [
                    "2009-12-07",
                    "2009-12-21",
                    "2009-12-07",
                    "2009-12-21",
                    "2009-12-07",
                ]
            ),
            "units_sold": [2, 3, 4, 5, 100],
        }
    )
    original = weekly.copy(deep=True)

    result = build_weekly_panel(
        weekly,
        product_codes=["A", "B"],
        start=pd.Timestamp("2009-12-07"),
        end_exclusive=pd.Timestamp("2009-12-28"),
    )

    expected = pd.DataFrame(
        {
            "StockCode": ["A", "A", "A", "B", "B", "B"],
            "week_start": pd.to_datetime(
                [
                    "2009-12-07",
                    "2009-12-14",
                    "2009-12-21",
                    "2009-12-07",
                    "2009-12-14",
                    "2009-12-21",
                ]
            ),
            "units_sold": pd.Series([2, None, 3, 4, None, 5], dtype="Float64"),
        }
    )

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(weekly, original)
