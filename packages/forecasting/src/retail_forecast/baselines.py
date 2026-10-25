import pandas as pd


def predict_previous_week(panel: pd.DataFrame) -> pd.DataFrame:
    """Predict each week's sales using the previous calendar week's sales."""
    result = panel.sort_values(["StockCode", "week_start"]).copy()

    if result.duplicated(["StockCode", "week_start"]).any():
        raise ValueError("Each product-week must be unique.")

    previous_dates = result.groupby("StockCode")["week_start"].shift(1)
    gaps = result["week_start"] - previous_dates

    if gaps.dropna().ne(pd.Timedelta(days=7)).any():
        raise ValueError("Each product must have a continuous weekly calendar.")

    result["prediction"] = result.groupby("StockCode")["units_sold"].shift(1)

    return result.reset_index(drop=True)


def predict_trailing_mean(
    panel: pd.DataFrame,
    window: int = 4,
) -> pd.DataFrame:
    """Predict using the previous complete window of weekly sales."""
    if window < 1:
        raise ValueError("Window must be positive.")

    result = predict_previous_week(panel)

    result["prediction"] = (
        result.groupby("StockCode")["prediction"]
        .transform(
            lambda values: values.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )
        .astype("Float64")
    )

    return result
