import argparse
from pathlib import Path

import pandas as pd


def inspect_overlap(path: Path) -> None:
    """Compare distinct transaction rows in the overlapping date period."""
    overlapping_data = []

    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        for sheet_name in ("Year 2009-2010", "Year 2010-2011"):
            df = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                dtype={"Invoice": "string", "StockCode": "string"},
            )

            mask = (df["InvoiceDate"] >= "2010-12-01") & (df["InvoiceDate"] < "2010-12-10")
            overlap = df.loc[mask]

            print(f"{sheet_name}: {len(overlap):,} rows in overlap period")
            overlapping_data.append(overlap.drop_duplicates())

    comparison = overlapping_data[0].merge(
        overlapping_data[1],
        on=overlapping_data[0].columns.tolist(),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    print("\nDistinct row comparison:")
    print(comparison["_merge"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check retail workbook overlap.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    if not args.path.is_file():
        parser.error(f"File not found: {args.path}")

    inspect_overlap(args.path)


if __name__ == "__main__":
    main()
