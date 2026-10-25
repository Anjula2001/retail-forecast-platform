import pandas as pd
from retail_forecast.ingestion import load_transactions


def test_load_transactions_uses_sheet_boundary_and_preserves_duplicates(tmp_path):
    path = tmp_path / "retail.xlsx"

    earlier = pd.DataFrame(
        {
            "Invoice": ["0001", "0001", "0002"],
            "StockCode": ["001A", "001A", "002B"],
            "InvoiceDate": pd.to_datetime(["2010-11-30", "2010-11-30", "2010-12-01"]),
        }
    )
    later = pd.DataFrame(
        {
            "Invoice": ["0003", "0004", "0005"],
            "StockCode": ["003C", "004D", "005E"],
            "InvoiceDate": pd.to_datetime(["2010-11-30", "2010-12-01", "2010-12-02"]),
        }
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        earlier.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        later.to_excel(writer, sheet_name="Year 2010-2011", index=False)

    result = load_transactions(path)

    assert result["Invoice"].tolist() == ["0001", "0001", "0004", "0005"]
    assert result["StockCode"].tolist() == ["001A", "001A", "004D", "005E"]
    assert result.index.tolist() == [0, 1, 2, 3]
