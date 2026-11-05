import hashlib
import json
import platform
import tomllib
from importlib.metadata import version
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd

from retail_forecast.evaluation import evaluate_forecasts
from retail_forecast.features import add_lag_features
from retail_forecast.training import (
    FEATURE_COLUMNS,
    train_random_forest,
    train_random_forest_with_product_id,
)


def main() -> None:
    config_path = Path("configs/forecast.toml")
    dataset_path = Path("data/processed/weekly_sales.csv")

    with config_path.open("rb") as file:
        split = tomllib.load(file)["split"]

    train_start = pd.Timestamp(split["train_start"])
    validation_start = pd.Timestamp(split["validation_start"])
    validation_end = pd.Timestamp(split["test_start"])

    panel = pd.read_csv(
        dataset_path,
        dtype={"StockCode": "string", "units_sold": "Float64"},
        parse_dates=["week_start"],
    )

    # Keep training history for lags and reserve the final test period.
    features = add_lag_features(panel.loc[panel["week_start"].lt(validation_end)])
    validation = features.loc[features["week_start"].ge(validation_start)].dropna(
        subset=["StockCode", "units_sold", *FEATURE_COLUMNS]
    )

    if validation.empty:
        raise ValueError("No usable validation rows.")

    rf = train_random_forest(features, train_start, validation_start)
    rf_with_id = train_random_forest_with_product_id(features, train_start, validation_start)

    predictions = {
        "mean_4_weeks": validation["rolling_mean_4"],
        "random_forest": rf.predict(validation[list(FEATURE_COLUMNS)].astype("float64")),
        "random_forest_with_product_id": rf_with_id.predict(
            validation[["StockCode", *FEATURE_COLUMNS]]
        ),
    }

    metrics = []
    for name, values in predictions.items():
        scoring = validation[["units_sold"]].assign(prediction=values)
        metrics.append({"method": name, **evaluate_forecasts(scoring)})

    output_dir = Path("artifacts/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="run-", dir=output_dir))
    report_path = run_dir / "metrics.json"
    report_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "train_start": str(train_start.date()),
        "validation_start": str(validation_start.date()),
        "validation_end_exclusive": str(validation_end.date()),
        "validation_scored_rows": len(validation),
        "feature_columns": list(FEATURE_COLUMNS),
        "models": {
            "mean_4_weeks": {"window": 4},
            "random_forest": rf.get_params(deep=False),
            "random_forest_with_product_id": {
                "categorical_feature": "StockCode",
                "encoder": "one_hot",
                "forest": rf_with_id.named_steps["model"].get_params(deep=False),
            },
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
    }

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Run metadata: {metadata_path}")

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
