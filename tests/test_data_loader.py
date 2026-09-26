import os
import pandas as pd
import pytest
from src.data_loader import load_raw_data, clean_data

def test_load_raw_data_success(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("ID,Year_Birth,Income,Dt_Customer\n1,1980,50000,01-01-2014\n")
    df = load_raw_data(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Income" in df.columns

def test_clean_data_drops_null_income():
    raw_df = pd.DataFrame({
        "ID": [1, 2],
        "Year_Birth": [1980, 1990],
        "Income": [50000.0, None],
        "Dt_Customer": ["01-01-2014", "02-02-2014"]
    })
    cleaned = clean_data(raw_df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["ID"] == 1
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Dt_Customer"])
