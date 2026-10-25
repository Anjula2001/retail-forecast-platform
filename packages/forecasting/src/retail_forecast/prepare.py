import argparse
import tomllib
from pathlib import Path

import pandas as pd

from retail_forecast.aggregation import (
    aggregate_observed_weekly_sales,
    build_weekly_panel,
    select_full_calendar_weeks,
)
from retail_forecast.cleaning import (
    select_merchandise_sales,
    select_sales_candidates,
)
from retail_forecast.ingestion import load_transactions
from retail_forecast.selection import select_training_products


def prepare_weekly_data(
    input_path: Path,
    config_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the weekly panel and training-selected product summary."""
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    split = config["split"]
    selection = config["selection"]

    train_start = pd.Timestamp(split["train_start"])
    validation_start = pd.Timestamp(split["validation_start"])
    test_start = pd.Timestamp(split["test_start"])
    end_exclusive = pd.Timestamp(split["end_exclusive"])

    if not train_start < validation_start < test_start < end_exclusive:
        raise ValueError("Split boundaries must be in chronological order.")

    raw = load_transactions(input_path)
    sales = select_merchandise_sales(select_sales_candidates(raw))
    weekly = aggregate_observed_weekly_sales(sales)

    weekly = select_full_calendar_weeks(
        weekly,
        coverage_start=train_start,
        coverage_end_exclusive=end_exclusive,
    )

    selected = select_training_products(
        weekly,
        train_start=train_start,
        train_end_exclusive=validation_start,
        n_products=selection["n_products"],
        min_sales_weeks=selection["min_sales_weeks"],
    )

    panel = build_weekly_panel(
        weekly,
        product_codes=selected["StockCode"].tolist(),
        start=train_start,
        end_exclusive=end_exclusive,
    )

    return panel, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare weekly retail sales.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.input, args.config):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    panel, selected = prepare_weekly_data(args.input, args.config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output_dir / "weekly_sales.csv", index=False)
    selected.to_csv(args.output_dir / "selected_products.csv", index=False)

    print(f"Products: {len(selected)}")
    print(f"Product-week rows: {len(panel)}")
    print(f"Missing sales: {panel['units_sold'].isna().sum()}")
    print(f"Saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
