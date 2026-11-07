import argparse
import hashlib
import json
import platform
import tomllib
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from retail_forecast.evaluation import evaluate_forecasts
from retail_forecast.features import add_lag_features
from retail_forecast.training import (
    FEATURE_COLUMNS,
    train_random_forest,
    train_random_forest_with_product_id,
)


def file_sha256(path: Path) -> str:
    """Identify the exact input file used for a run."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run_experiment(dataset_path: Path, config_path: Path, output_dir: Path) -> Path:
    """Train and compare validation candidates, saving a reproducible run bundle."""
    with config_path.open("rb") as file:
        split = tomllib.load(file)["split"]

    train_start = pd.Timestamp(split["train_start"])
    validation_start = pd.Timestamp(split["validation_start"])
    validation_end = pd.Timestamp(split["test_start"])
    if not train_start < validation_start < validation_end:
        raise ValueError("Training and validation boundaries must be chronological.")

    panel = pd.read_csv(
        dataset_path,
        dtype={"StockCode": "string", "units_sold": "Float64"},
        parse_dates=["week_start"],
    )

    # Keep the test period out of feature generation and candidate comparison.
    features = add_lag_features(panel.loc[panel["week_start"].lt(validation_end)])
    validation = features.loc[features["week_start"].ge(validation_start)].dropna(
        subset=["StockCode", "units_sold", *FEATURE_COLUMNS]
    )
    if validation.empty:
        raise ValueError("No usable validation rows.")

    random_forest = train_random_forest(features, train_start, validation_start)
    random_forest_with_product_id = train_random_forest_with_product_id(
        features, train_start, validation_start
    )
    prediction_frame = validation[["StockCode", "week_start", "units_sold"]].copy()
    prediction_frame["mean_4_weeks"] = validation["rolling_mean_4"]
    prediction_frame["random_forest"] = random_forest.predict(
        validation[list(FEATURE_COLUMNS)].astype("float64")
    )
    prediction_frame["random_forest_with_product_id"] = random_forest_with_product_id.predict(
        validation[["StockCode", *FEATURE_COLUMNS]]
    )

    candidate_names = [
        "mean_4_weeks",
        "random_forest",
        "random_forest_with_product_id",
    ]
    metrics = []
    for name in candidate_names:
        scoring = prediction_frame[["units_sold", name]].rename(columns={name: "prediction"})
        metrics.append({"method": name, **evaluate_forecasts(scoring)})

    run_id = f"run-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    predictions_path = run_dir / "validation_predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)
    models_path = run_dir / "models.joblib"
    joblib.dump(
        {
            "random_forest": random_forest,
            "random_forest_with_product_id": random_forest_with_product_id,
        },
        models_path,
    )

    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "evaluation": "rolling_one_week_ahead",
        "train_start": str(train_start.date()),
        "validation_start": str(validation_start.date()),
        "validation_end_exclusive": str(validation_end.date()),
        "validation_scored_rows": len(prediction_frame),
        "feature_columns": list(FEATURE_COLUMNS),
        "artifacts": {
            "metrics": metrics_path.name,
            "validation_predictions": predictions_path.name,
            "models": models_path.name,
        },
        "models": {
            "mean_4_weeks": {"window": 4},
            "random_forest": random_forest.get_params(deep=False),
            "random_forest_with_product_id": {
                "categorical_feature": "StockCode",
                "encoder": "one_hot",
                "forest": random_forest_with_product_id.named_steps["model"].get_params(deep=False),
            },
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nSaved run bundle: {run_dir.resolve()}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare retail forecast candidates.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/weekly_sales.csv"),
        help="Prepared weekly sales CSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecast.toml"),
        help="Forecast split configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Parent directory for timestamped run bundles.",
    )
    args = parser.parse_args()

    for path in (args.input, args.config):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    run_experiment(args.input, args.config, args.output_dir)


if __name__ == "__main__":
    main()
