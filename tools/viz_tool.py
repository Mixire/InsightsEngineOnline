import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from loguru import logger
from config import CHARTS_DIR

os.makedirs(CHARTS_DIR, exist_ok=True)


def generate_all_charts(df: pd.DataFrame, target_col: str = None, task: str = None):
    """Generates all relevant charts based on data and task type."""
    logger.info("Generating visualizations...")

    _pairplot(df, target_col)

    if target_col and task in ["binary_classification", "multiclass_classification"]:
        _class_balance_chart(df, target_col)

    if target_col and task == "regression":
        _target_distribution(df, target_col)

    _boxplots(df)
    logger.success("All charts generated.")


def _pairplot(df: pd.DataFrame, target_col: str = None):
    numeric = df.select_dtypes(include="number")
    cols = [c for c in numeric.columns[:5]]
    if len(cols) < 2:
        return
    
    hue = target_col if target_col and target_col in df.columns else None
    
    # Ensure unique columns to avoid ValueError: Data must be 1-dimensional
    plot_cols = list(cols)
    if hue and hue not in plot_cols:
        plot_cols.append(hue)
    
    sns.pairplot(df[plot_cols], hue=hue, diag_kind="kde")
    plt.savefig(f"{CHARTS_DIR}pairplot.png", dpi=120)
    plt.close()
    logger.info("Saved: pairplot.png")


def _class_balance_chart(df: pd.DataFrame, target_col: str):
    plt.figure(figsize=(8, 4))
    df[target_col].value_counts().plot(kind="bar", color="#4F8EF7", edgecolor="black")
    plt.title(f"Class Balance: {target_col}")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}class_balance.png", dpi=150)
    plt.close()


def _target_distribution(df: pd.DataFrame, target_col: str):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[target_col].dropna(), kde=True, color="#27AE60")
    plt.title(f"Target Distribution: {target_col}")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}target_distribution.png", dpi=150)
    plt.close()


def _boxplots(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i], color="#9B59B6")
        axes[i].set_title(col)
    for j in range(i + 1, 6):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}boxplots.png", dpi=150)
    plt.close()
    logger.info("Saved: boxplots.png")
