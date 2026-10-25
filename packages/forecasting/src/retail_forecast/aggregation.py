import pandas as pd


def aggregate_observed_weekly_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Sum observed product sales into Monday-to-Sunday weeks."""
    data = sales[["StockCode", "InvoiceDate", "Quantity"]].copy()

    data["week_start"] = (
        pd.to_datetime(data["InvoiceDate"], errors="raise").dt.to_period("W-SUN").dt.start_time
    )

    return (
        data.groupby(["StockCode", "week_start"], as_index=False)
        .agg(units_sold=("Quantity", "sum"))
        .sort_values(["StockCode", "week_start"])
        .reset_index(drop=True)
    )


def select_full_calendar_weeks(
    weekly: pd.DataFrame,
    coverage_start: pd.Timestamp,
    coverage_end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Keep weeks fully inside the specified source coverage interval."""
    if coverage_end_exclusive <= coverage_start:
        raise ValueError("Coverage end must be after coverage start.")

    week_end_exclusive = weekly["week_start"] + pd.Timedelta(days=7)

    keep = (weekly["week_start"] >= coverage_start) & (week_end_exclusive <= coverage_end_exclusive)

    return weekly.loc[keep].copy().reset_index(drop=True)


def build_weekly_panel(
    weekly: pd.DataFrame,
    product_codes: list[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Create a product-week grid, preserving missing sales as unknown."""
    if end_exclusive <= start:
        raise ValueError("End must be after start.")

    if start.day_of_week != 0 or end_exclusive.day_of_week != 0:
        raise ValueError("Calendar boundaries must be Mondays.")

    weeks = pd.date_range(
        start=start,
        end=end_exclusive,
        freq="W-MON",
        inclusive="left",
    )
    grid = pd.MultiIndex.from_product(
        [product_codes, weeks],
        names=["StockCode", "week_start"],
    )

    return (
        weekly.set_index(["StockCode", "week_start"])["units_sold"]
        .astype("Float64")
        .reindex(grid)
        .reset_index()
    )
