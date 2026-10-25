import pandas as pd
from retail_forecast.cleaning import (
    select_merchandise_sales,
    select_sales_candidates,
)


def test_select_merchandise_sales_preserves_unusual_product_codes():
    sales = pd.DataFrame(
        {
            "StockCode": [
                "POST",
                "BANK CHARGES",
                " m ",
                "gift_0001_20",
                "TEST001",
                "DCGS0076",
                "SP1002",
                "PADS",
                "85123A",
            ]
        }
    )
    original = sales.copy(deep=True)

    result = select_merchandise_sales(sales)

    assert result["StockCode"].tolist() == [
        "DCGS0076",
        "SP1002",
        "PADS",
        "85123A",
    ]
    pd.testing.assert_frame_equal(sales, original)


def test_select_sales_candidates_filters_invalid_sales_without_mutating_input():
    transactions = pd.DataFrame(
        [
            ("0001", 2, 10),
            ("C0002", 2, 10),
            (" c0003 ", 2, 10),
            ("0004", 0, 10),
            ("0005", -1, 10),
            ("0006", 2, 0),
            ("0007", 2, -5),
            (None, 2, 10),
            ("   ", 2, 10),
            ("0008", None, 10),
            ("0009", 2, None),
        ],
        columns=["Invoice", "Quantity", "Price"],
    )
    transactions["Customer ID"] = None
    transactions["Description"] = None
    original = transactions.copy(deep=True)

    result = select_sales_candidates(transactions)

    assert result["Invoice"].tolist() == ["0001"]
    assert result["Customer ID"].isna().all()
    assert result["Description"].isna().all()
    pd.testing.assert_frame_equal(transactions, original)
