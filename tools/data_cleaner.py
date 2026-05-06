import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from loguru import logger


def auto_clean(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    """
    Automatically cleans a dataframe:
    - Drops columns with >70% null values
    - Fills numeric nulls with median
    - Fills categorical nulls with mode
    - Encodes categorical columns
    - Removes duplicate rows
    """
    original_shape = df.shape
    logger.info("Starting auto-clean...")

    # Drop high-null columns
    null_pct = df.isnull().mean()
    drop_cols = null_pct[null_pct > 0.7].index.tolist()
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        logger.warning(f"Dropped high-null columns: {drop_cols}")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill nulls (Fix for Pandas 3.0 FutureWarning)
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categoricals (skip target if classification)
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == target_col:
            continue
        if df[col].nunique() <= 20:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df.drop(columns=[col], inplace=True)
            logger.warning(f"Dropped high-cardinality column: {col}")

    logger.success(f"Cleaning done: {original_shape} → {df.shape}")
    return df


def encode_target(series: pd.Series) -> pd.Series:
    """Label encodes a target column."""
    le = LabelEncoder()
    return pd.Series(le.fit_transform(series.astype(str)), name=series.name)
