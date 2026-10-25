import pandas as pd


def summarize_training_products(
    weekly: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Summarize products using only training-period sales."""
    training = weekly.loc[
        (weekly["week_start"] >= train_start) & (weekly["week_start"] < train_end_exclusive)
    ]

    return (
        training.groupby("StockCode", as_index=False)
        .agg(
            sales_weeks=("week_start", "nunique"),
            total_units=("units_sold", "sum"),
            first_sales_week=("week_start", "min"),
            last_sales_week=("week_start", "max"),
        )
        .sort_values(
            ["sales_weeks", "total_units", "StockCode"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def select_training_products(
    weekly: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end_exclusive: pd.Timestamp,
    n_products: int,
    min_sales_weeks: int,
) -> pd.DataFrame:
    """Select regularly selling products using training data only."""
    if n_products < 1 or min_sales_weeks < 1:
        raise ValueError("Selection settings must be positive.")

    summary = summarize_training_products(
        weekly,
        train_start=train_start,
        train_end_exclusive=train_end_exclusive,
    )
    eligible = summary.loc[summary["sales_weeks"] >= min_sales_weeks]

    if len(eligible) < n_products:
        raise ValueError(f"Requested {n_products} products, but only {len(eligible)} qualify.")

    return eligible.head(n_products).copy().reset_index(drop=True)
