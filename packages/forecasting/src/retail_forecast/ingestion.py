from pathlib import Path

import pandas as pd

SHEET_BOUNDARY = pd.Timestamp("2010-12-01")


def load_transactions(path: Path) -> pd.DataFrame:
    """Load source transactions using non-overlapping sheet periods."""
    identifier_types = {"Invoice": "string", "StockCode": "string"}

    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        earlier = pd.read_excel(
            workbook,
            sheet_name="Year 2009-2010",
            dtype=identifier_types,
        )
        later = pd.read_excel(
            workbook,
            sheet_name="Year 2010-2011",
            dtype=identifier_types,
        )

    earlier = earlier.loc[earlier["InvoiceDate"] < SHEET_BOUNDARY]
    later = later.loc[later["InvoiceDate"] >= SHEET_BOUNDARY]

    return pd.concat([earlier, later], ignore_index=True)
