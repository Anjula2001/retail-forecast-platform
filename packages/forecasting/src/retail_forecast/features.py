import pandas as pd


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add past weekly sales without using current-week targets."""
    result = panel.sort_values(["StockCode", "week_start"]).copy()

    if result.duplicated(["StockCode", "week_start"]).any():
        raise ValueError("Each product-week must be unique.")

    previous_dates = result.groupby("StockCode")["week_start"].shift(1)
    gaps = result["week_start"] - previous_dates

    if gaps.dropna().ne(pd.Timedelta(days=7)).any():
        raise ValueError("Each product must have a continuous weekly calendar.")

    grouped_sales = result.groupby("StockCode")["units_sold"]

    for lag in (1, 2, 4):
        result[f"lag_{lag}"] = grouped_sales.shift(lag)

    result["rolling_mean_4"] = (
        result.groupby("StockCode")["lag_1"]
        .transform(
            lambda values: values.rolling(
                window=4,
                min_periods=4,
            ).mean()
        )
        .astype("Float64")
    )

    return result.reset_index(drop=True)
