import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import pandas as pd

from retail_forecast.baselines import (
    predict_previous_week,
    predict_trailing_mean,
)
from retail_forecast.evaluation import evaluate_forecasts


def file_sha256(path: Path) -> str:
    """Identify the exact input file used for a run."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validation baselines.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.input, args.config):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    with args.config.open("rb") as file:
        split = tomllib.load(file)["split"]

    validation_start = pd.Timestamp(split["validation_start"])
    validation_end = pd.Timestamp(split["test_start"])

    if validation_end <= validation_start:
        parser.error("Validation end must be after validation start.")

    panel = pd.read_csv(
        args.input,
        dtype={"StockCode": "string", "units_sold": "Float64"},
        parse_dates=["week_start"],
    )

    forecasts = {
        "previous_week": predict_previous_week(panel),
        "mean_4_weeks": predict_trailing_mean(panel, window=4),
    }

    validation = panel.loc[
        panel["week_start"].ge(validation_start) & panel["week_start"].lt(validation_end)
    ].copy()

    keys = ["StockCode", "week_start"]

    for name, frame in forecasts.items():
        predictions = frame[keys + ["prediction"]].rename(columns={"prediction": name})
        validation = validation.merge(
            predictions,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    names = list(forecasts)
    common = validation[["units_sold", *names]].notna().all(axis=1)
    metrics = []

    for name in names:
        scoring = validation.loc[common, ["units_sold", name]].rename(columns={name: "prediction"})
        metrics.append({"baseline": name, **evaluate_forecasts(scoring)})

    report = {
        "schema_version": 1,
        "evaluation": "rolling_one_week_ahead",
        "validation_start": str(validation_start.date()),
        "validation_end_exclusive": str(validation_end.date()),
        "validation_rows": len(validation),
        "common_scored_rows": int(common.sum()),
        "input_sha256": file_sha256(args.input),
        "config_sha256": file_sha256(args.config),
        "metrics": metrics,
    }

    validation["scored"] = common
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation.to_csv(
        args.output_dir / "validation_predictions.csv",
        index=False,
    )
    report_path = args.output_dir / "validation_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nSaved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
