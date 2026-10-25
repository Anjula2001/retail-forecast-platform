import pandas as pd


def evaluate_forecasts(forecasts: pd.DataFrame) -> dict[str, int | float | None]:
    """Evaluate rows with both known actual sales and predictions."""
    actual = forecasts["units_sold"]
    prediction = forecasts["prediction"]
    paired = actual.notna() & prediction.notna()

    if not paired.any():
        raise ValueError("No paired actuals and predictions to evaluate.")

    errors = (actual.loc[paired] - prediction.loc[paired]).abs()
    actual_total = actual.loc[paired].abs().sum()

    return {
        "rows": len(forecasts),
        "known_actuals": int(actual.notna().sum()),
        "scored_rows": int(paired.sum()),
        "mae": float(errors.mean()),
        "wape_pct": (float(errors.sum() / actual_total * 100) if actual_total > 0 else None),
    }
