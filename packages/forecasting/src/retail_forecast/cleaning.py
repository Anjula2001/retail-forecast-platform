import pandas as pd

EXCLUDED_STOCK_CODES = frozenset(
    {
        "POST",
        "DOT",
        "C2",
        "BANK CHARGES",
        "AMAZONFEE",
        "D",
        "B",
        "ADJUST",
        "ADJUST2",
        "TEST001",
        "TEST002",
        "M",
        "S",
    }
)


def select_sales_candidates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Select positive, paid, non-cancelled transaction lines."""
    invoices = transactions["Invoice"].astype("string").str.strip()

    is_cancelled = invoices.str.upper().str.startswith("C", na=False)
    has_positive_quantity = transactions["Quantity"].gt(0)
    has_positive_price = transactions["Price"].gt(0)

    keep = (
        invoices.notna()
        & invoices.ne("")
        & ~is_cancelled
        & has_positive_quantity
        & has_positive_price
    )

    return transactions.loc[keep].copy()


def select_merchandise_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Exclude reviewed charges, vouchers and out-of-scope entries."""
    codes = sales["StockCode"].astype("string").str.strip().str.upper()

    excluded = codes.isin(EXCLUDED_STOCK_CODES) | codes.str.startswith("GIFT_0001_", na=False)

    return sales.loc[~excluded].copy()
0000000