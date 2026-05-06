import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error,
    r2_score, silhouette_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from loguru import logger
import matplotlib.pyplot as plt
import os
from config import CHARTS_DIR

os.makedirs(CHARTS_DIR, exist_ok=True)


ALGORITHM_MAP = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoostRegressor": XGBRegressor(random_state=42),
    "LGBMRegressor": LGBMRegressor(random_state=42),
    "KMeans": KMeans(n_clusters=4, random_state=42, n_init="auto"),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=4),
    "IsolationForest": IsolationForest(random_state=42),
    "LocalOutlierFactor": LocalOutlierFactor(),
}


def run_model(df: pd.DataFrame, task: str, algorithm: str,
              target_col: str, feature_cols: list) -> dict:
    """
    Trains and evaluates the specified model.
    Returns metrics, feature importance, and model object.
    """
    logger.info(f"Running: {algorithm} for task: {task}")
    model = ALGORITHM_MAP.get(algorithm)
    if model is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    X = df[feature_cols].select_dtypes(include="number")
    feature_cols = list(X.columns)

    if task in ["binary_classification", "multiclass_classification"]:
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds, average="weighted"), 4),
            "report": classification_report(y_test, preds, output_dict=True)
        }
        feature_importance = _get_feature_importance(model, feature_cols)
        _plot_feature_importance(feature_importance, algorithm)

    elif task == "regression":
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "r2_score": round(r2_score(y_test, preds), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            "mae": round(np.mean(np.abs(y_test - preds)), 4)
        }
        feature_importance = _get_feature_importance(model, feature_cols)
        _plot_feature_importance(feature_importance, algorithm)

    elif task == "clustering":
        model.fit(X)
        labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
        metrics = {
            "silhouette_score": round(score, 4),
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)
        }
        feature_importance = {}

    elif task == "anomaly_detection":
        model.fit(X)
        preds = model.predict(X) if hasattr(model, "predict") else model.fit_predict(X)
        n_anomalies = int((preds == -1).sum())
        metrics = {
            "anomalies_detected": n_anomalies,
            "anomaly_rate": round(n_anomalies / len(X), 4)
        }
        feature_importance = {}

    else:
        raise ValueError(f"Unsupported task: {task}")

    logger.success(f"Model complete. Metrics: {metrics}")
    return {"metrics": metrics, "feature_importance": feature_importance, "model": model}


def _get_feature_importance(model, feature_cols: list) -> dict:
    if hasattr(model, "feature_importances_"):
        return dict(sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        ))
    elif hasattr(model, "coef_"):
        coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        return dict(sorted(
            zip(feature_cols, abs(coef)),
            key=lambda x: x[1], reverse=True
        ))
    return {}


def _plot_feature_importance(importance: dict, algorithm: str):
    if not importance:
        return
    top = dict(list(importance.items())[:15])
    plt.figure(figsize=(10, 6))
    plt.barh(list(top.keys()), list(top.values()), color="#4F8EF7")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance — {algorithm}")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}feature_importance.png", dpi=150)
    plt.close()
    logger.info("Saved: feature_importance.png")
