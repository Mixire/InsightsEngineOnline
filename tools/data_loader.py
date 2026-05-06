import pandas as pd
import json
from sqlalchemy import create_engine
from loguru import logger

def load_data(source: str, db_query: str = None, db_connection: str = None) -> pd.DataFrame:
    """
    Load data from CSV, Excel, JSON, or SQL database.
    Returns a pandas DataFrame.
    """
    logger.info(f"Loading data from: {source}")

    if db_connection and db_query:
        engine = create_engine(db_connection)
        df = pd.read_sql(db_query, engine)

    elif source.endswith(".csv"):
        df = pd.read_csv(source)

    elif source.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)

    elif source.endswith(".json"):
        df = pd.read_json(source)

    else:
        raise ValueError(f"Unsupported file format: {source}")

    logger.success(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Returns a structured summary of the dataframe for the agent."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentage": (df.isnull().mean() * 100).round(2).to_dict(),
        "sample_rows": df.head(3).to_dict(orient="records"),
        "numeric_stats": df.describe().to_dict(),
        "unique_counts": df.nunique().to_dict(),
    }
