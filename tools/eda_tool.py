import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from loguru import logger
from config import CHARTS_DIR

os.makedirs(CHARTS_DIR, exist_ok=True)


def run_eda(df: pd.DataFrame) -> dict:
    """Runs automated EDA and saves charts. Returns a summary dict."""
    logger.info("Running EDA...")
    results = {}

    # Basic stats
    results["shape"] = df.shape
    results["describe"] = df.describe().to_dict()
    results["correlation"] = df.select_dtypes(include="number").corr().to_dict()
    results["top_correlations"] = _get_top_correlations(df)

    # Charts
    _plot_distributions(df)
    _plot_correlation_heatmap(df)
    _plot_missing_values(df)

    logger.success("EDA complete.")
    return results


def _get_top_correlations(df: pd.DataFrame, top_n: int = 10) -> list:
    corr = df.select_dtypes(include="number").corr().abs()
    # Fix for pandas removal of pd.np
    mask = np.tril(np.ones(corr.shape)).astype(bool)
    pairs = (
        corr.where(~mask)
        .stack()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return [(str(idx), round(val, 3)) for idx, val in pairs.items()]


def _plot_distributions(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns[:9]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), ax=axes[i], kde=True, color="#4F8EF7")
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, 9):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}distributions.png", dpi=150)
    plt.close()
    logger.info("Saved: distributions.png")


def _plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}correlation_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved: correlation_heatmap.png")


def _plot_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return
    plt.figure(figsize=(10, 4))
    missing.sort_values().plot(kind="barh", color="#FF6B6B")
    plt.title("Missing Values per Column")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}missing_values.png", dpi=150)
    plt.close()
    logger.info("Saved: missing_values.png")
