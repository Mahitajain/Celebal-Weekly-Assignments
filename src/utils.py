"""
utils.py – Shared helpers for the Tesla ML pipeline.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(ROOT, "data", "raw", "tesla_deliveries_dataset_2015_2025.csv")
DATA_PROC   = os.path.join(ROOT, "data", "processed_data.csv")
MODELS_DIR  = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
REPORTS_DIR = os.path.join(ROOT, "reports")

for d in [MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, os.path.join(ROOT, "data")]:
    os.makedirs(d, exist_ok=True)


# ── Metrics ────────────────────────────────────────────────────────────────────
def regression_metrics(y_true, y_pred, label: str = "") -> dict:
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        mean_absolute_percentage_error,
    )
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    if label:
        print(f"  [{label}]  MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return dict(label=label, MAE=mae, RMSE=rmse, R2=r2, MAPE=mape)


# ── Persistence ────────────────────────────────────────────────────────────────
def save_model(model, name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  ✓ Model saved → {path}")
    return path


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)


def save_metrics(metrics_list: list, name: str = "all_metrics"):
    path = os.path.join(REPORTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"  ✓ Metrics saved → {path}")


# ── Plotting ───────────────────────────────────────────────────────────────────
PALETTE = ["#E31937", "#1B1B1B", "#5B9BD5", "#70AD47", "#FFC000", "#7030A0"]

def savefig(name: str, tight=True):
    path = os.path.join(OUTPUTS_DIR, f"{name}.png")
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  ✓ Plot saved  → {path}")
    return path


def section(title: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")
