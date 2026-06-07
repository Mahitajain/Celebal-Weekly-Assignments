"""
04_regression_modeling.py
─────────────────────────────────────────────────────────────────
Stage 4 – Regression Modeling

Models trained
──────────────
1. Linear Regression (baseline)
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. XGBoost Regressor

Pipeline includes StandardScaler → Model.
80/20 chronological split (no shuffle to respect time ordering).

Outputs
───────
• outputs/model_comparison.png     – R² & RMSE comparison bar charts
• outputs/residuals_*.png          – Residual plots for top 3 models
• outputs/actual_vs_pred_*.png     – Actual vs Predicted scatter
• models/*.pkl                     – All trained pipelines
• reports/model_metrics.json       – All metrics
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LinearRegression, Ridge, Lasso
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from xgboost                 import XGBRegressor
from sklearn.model_selection import cross_val_score

from utils import (DATA_PROC, OUTPUTS_DIR, MODELS_DIR,
                   regression_metrics, save_model, save_metrics, savefig, section, PALETTE)

FEATURES_PATH = os.path.join(os.path.dirname(DATA_PROC), "features.csv")

TARGET = "Estimated_Deliveries"


# ─────────────────────────────────────────────────────────────────────────────
def _build_xy(df):
    """Return X (feature matrix) and y (target) with sensible column selection."""
    exclude = {TARGET, "log_Deliveries", "Date", "Year", "Month"}   # keep Year_progress etc.
    feat_cols = [c for c in df.columns
                 if c not in exclude
                 and df[c].dtype in [np.float64, np.int64, int, float]]
    X = df[feat_cols].fillna(0)
    y = df[TARGET]
    return X, y, feat_cols


def run():
    section("STAGE 4 – REGRESSION MODELING")

    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    X, y, feat_cols = _build_xy(df)
    print(f"\n  Feature matrix: {X.shape}  |  Target range: {y.min():,} – {y.max():,}")

    # ── Chronological train/test split ───────────────────────────────────────
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # ── Model zoo ────────────────────────────────────────────────────────────
    models = {
        "Linear Regression":    Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Ridge":                Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "Lasso":                Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=50.0, max_iter=5000))]),
        "Random Forest":        RandomForestRegressor(n_estimators=200, max_depth=12,
                                                      min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                          max_depth=5, subsample=0.85, random_state=42),
        "XGBoost":              XGBRegressor(n_estimators=300, learning_rate=0.07, max_depth=6,
                                             subsample=0.85, colsample_bytree=0.8,
                                             verbosity=0, random_state=42, n_jobs=-1),
    }

    all_metrics = []
    predictions = {}

    print()
    for name, model in models.items():
        print(f"  Training: {name} …")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)

        m = regression_metrics(y_test, preds, label=name)
        all_metrics.append(m)
        predictions[name] = preds

        # 5-fold CV on train set
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5,
                                scoring="r2", n_jobs=-1).mean()
        print(f"       CV R² (5-fold, train): {cv_r2:.4f}")

        save_model(model, name.replace(" ", "_"))

    # ── Model comparison charts ───────────────────────────────────────────────
    print("\n[Plot] Model comparison …")
    names    = [m["label"] for m in all_metrics]
    r2_vals  = [m["R2"]   for m in all_metrics]
    rmse_vals= [m["RMSE"] for m in all_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison – Test Set", fontsize=13, fontweight="bold")

    colors = [PALETTE[0] if v == max(r2_vals) else PALETTE[1] for v in r2_vals]
    axes[0].barh(names, r2_vals, color=colors, edgecolor="white")
    axes[0].set_title("R² Score (higher = better)")
    axes[0].set_xlim(0, 1.05)
    for i, v in enumerate(r2_vals):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    colors2 = [PALETTE[0] if v == min(rmse_vals) else PALETTE[1] for v in rmse_vals]
    axes[1].barh(names, rmse_vals, color=colors2, edgecolor="white")
    axes[1].set_title("RMSE (lower = better)")
    for i, v in enumerate(rmse_vals):
        axes[1].text(v + 10, i, f"{v:,.0f}", va="center", fontsize=9)

    savefig("model_comparison")

    # ── Feature importance (tree models) ────────────────────────────────────
    print("[Plot] Feature importance (Random Forest) …")
    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=feat_cols)
    top20 = importances.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    top20.plot(kind="barh", ax=ax, color=PALETTE[0], edgecolor="white")
    ax.set_title("Random Forest – Top 20 Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance")
    savefig("feature_importance")

    # ── Residual plots for top 3 ─────────────────────────────────────────────
    top3 = sorted(all_metrics, key=lambda x: x["R2"], reverse=True)[:3]
    for m in top3:
        name  = m["label"]
        preds = predictions[name]
        resid = y_test.values - preds

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Residual Analysis – {name}", fontweight="bold")

        axes[0].scatter(preds, resid, alpha=0.35, s=12, color=PALETTE[0])
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Fitted")

        axes[1].hist(resid, bins=40, color=PALETTE[2], edgecolor="white", linewidth=0.4)
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual")

        savefig(f"residuals_{name.replace(' ', '_')}")

    # ── Actual vs Predicted (best model) ─────────────────────────────────────
    best = max(all_metrics, key=lambda x: x["R2"])
    name = best["label"]
    preds = predictions[name]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, preds, alpha=0.35, s=15, color=PALETTE[0], label="Predictions")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect fit")
    ax.set_title(f"Actual vs Predicted – {name}\nR²={best['R2']:.4f}", fontweight="bold")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend()
    savefig(f"actual_vs_pred_{name.replace(' ', '_')}")

    # ── Save metrics JSON ────────────────────────────────────────────────────
    save_metrics(all_metrics, "model_metrics")

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n  🏆 Best model on test set: {best['label']}  |  R²={best['R2']:.4f}  RMSE={best['RMSE']:,.0f}")

    return all_metrics, best


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
