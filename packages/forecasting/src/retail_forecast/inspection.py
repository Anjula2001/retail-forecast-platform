import argparse
from pathlib import Path

import pandas as pd


def inspect_workbook(path: Path) -> None:
    """Print a basic quality summary for each workbook sheet."""
    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        for sheet_name in workbook.sheet_names:
            df = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                dtype={"Invoice": "string", "StockCode": "string"},
            )

            dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            cancellations = df["Invoice"].str.upper().str.startswith("C", na=False)

            print(f"\n=== {sheet_name} ===")
            print(f"Rows: {len(df):,}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Date range: {dates.min()} -> {dates.max()}")
            print(f"Missing or invalid dates: {dates.isna().sum():,}")
            print(f"Exact duplicate rows: {df.duplicated().sum():,}")
            print(f"Cancellation lines: {cancellations.sum():,}")
            print(f"Quantity <= 0: {df['Quantity'].le(0).sum():,}")
            print(f"Price <= 0: {df['Price'].le(0).sum():,}")
            print("\nMissing values by column:")
            print(df.isna().sum().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a retail Excel workbook.")
    parser.add_argument("path", type=Path, help="Path to the source Excel file.")
    args = parser.parse_args()

    if not args.path.is_file():
        parser.error(f"File not found: {args.path}")

    inspect_workbook(args.path)


if __name__ == "__main__":
    main()
