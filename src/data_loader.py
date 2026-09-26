import pandas as pd

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw customer dataset from CSV file."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize data: drop null income records and parse customer enrollment dates."""
    df_clean = df.copy()
    df_clean.dropna(subset=['Income'], inplace=True)
    df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], dayfirst=True)
    return df_clean
