"""
05_hyperparameter_tuning.py
─────────────────────────────────────────────────────────────────
Stage 5 – Hyperparameter Tuning

Strategy
────────
• Run RandomizedSearchCV on XGBoost (best tree model in Stage 4)
• 50 iterations, 5-fold time-series-aware CV (TimeSeriesSplit)
• After best params found, run a final fine-grain GridSearchCV
  on a narrow neighbourhood of the best params
• Save best model as models/best_model.pkl

Outputs
───────
• outputs/tuning_cv_scores.png   – CV score distribution across iterations
• outputs/learning_curve.png     – Bias–variance learning curve
• models/best_model.pkl          – Best tuned XGBoost
• reports/tuning_results.json    – Best params + scores
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection  import (RandomizedSearchCV, GridSearchCV,
                                       TimeSeriesSplit, learning_curve)
from sklearn.metrics          import make_scorer, r2_score
from xgboost                  import XGBRegressor

from utils import (DATA_PROC, REPORTS_DIR, savefig, save_model,
                   regression_metrics, section, PALETTE)

FEATURES_PATH = os.path.join(os.path.dirname(DATA_PROC), "features.csv")
TARGET = "Estimated_Deliveries"


# ─────────────────────────────────────────────────────────────────────────────
def _build_xy(df):
    exclude = {TARGET, "log_Deliveries", "Date", "Year", "Month"}
    feat_cols = [c for c in df.columns
                 if c not in exclude
                 and df[c].dtype in [np.float64, np.int64, int, float]]
    return df[feat_cols].fillna(0), df[TARGET], feat_cols


def run():
    section("STAGE 5 – HYPERPARAMETER TUNING")

    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    X, y, feat_cols = _build_xy(df)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    tscv = TimeSeriesSplit(n_splits=5)
    scorer = make_scorer(r2_score)

    # ── Phase 1: RandomizedSearchCV ──────────────────────────────────────────
    print("\n[Phase 1] RandomizedSearchCV on XGBoost (50 iterations) …")
    param_dist = {
        "n_estimators"    : [100, 200, 300, 400, 500],
        "learning_rate"   : [0.01, 0.03, 0.05, 0.07, 0.10, 0.15],
        "max_depth"       : [3, 4, 5, 6, 7, 8],
        "subsample"       : [0.6, 0.7, 0.8, 0.85, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma"           : [0, 0.1, 0.2, 0.5, 1.0],
        "reg_alpha"       : [0, 0.01, 0.1, 1.0],
        "reg_lambda"      : [1.0, 1.5, 2.0, 5.0],
    }

    xgb_base = XGBRegressor(verbosity=0, random_state=42, n_jobs=-1)
    rscv = RandomizedSearchCV(
        xgb_base, param_distributions=param_dist,
        n_iter=50, cv=tscv, scoring=scorer,
        n_jobs=-1, random_state=42, verbose=1,
        return_train_score=True,
    )
    rscv.fit(X_train, y_train)

    print(f"\n  Best CV R² (RandomizedSearch): {rscv.best_score_:.4f}")
    print(f"  Best params: {rscv.best_params_}")

    # ── Plot CV score distribution ───────────────────────────────────────────
    cv_results = pd.DataFrame(rscv.cv_results_)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cv_results["mean_test_score"], color=PALETTE[0], linewidth=1.5, label="Val R²")
    ax.plot(cv_results["mean_train_score"], color=PALETTE[2], linewidth=1.5,
            linestyle="--", label="Train R²")
    ax.axhline(rscv.best_score_, color="black", linewidth=1, linestyle=":")
    ax.set_title("RandomizedSearchCV – CV Score per Iteration", fontweight="bold")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean R² (5-fold)")
    ax.legend()
    savefig("tuning_cv_scores")

    # ── Phase 2: Narrow GridSearchCV around best params ──────────────────────
    print("\n[Phase 2] Fine-grained GridSearchCV …")
    bp = rscv.best_params_
    narrow_grid = {
        "n_estimators" : sorted({max(50, bp["n_estimators"] - 100),
                                  bp["n_estimators"],
                                  bp["n_estimators"] + 100}),
        "learning_rate": sorted({max(0.005, bp["learning_rate"] - 0.02),
                                  bp["learning_rate"],
                                  bp["learning_rate"] + 0.02}),
        "max_depth"    : sorted({max(2, bp["max_depth"] - 1),
                                  bp["max_depth"],
                                  bp["max_depth"] + 1}),
    }
    static_params = {k: v for k, v in bp.items()
                     if k not in narrow_grid}

    xgb_narrow = XGBRegressor(**static_params, verbosity=0, random_state=42, n_jobs=-1)
    gscv = GridSearchCV(
        xgb_narrow, narrow_grid,
        cv=tscv, scoring=scorer, n_jobs=-1, verbose=1,
        return_train_score=True,
    )
    gscv.fit(X_train, y_train)

    print(f"\n  Best CV R² (GridSearch): {gscv.best_score_:.4f}")
    print(f"  Best params: {gscv.best_params_}")

    best_model  = gscv.best_estimator_
    best_params = {**static_params, **gscv.best_params_}

    # ── Evaluate on test set ─────────────────────────────────────────────────
    preds = np.clip(best_model.predict(X_test), 0, None)
    final_metrics = regression_metrics(y_test, preds, label="Tuned XGBoost (test)")

    # ── Learning curve ───────────────────────────────────────────────────────
    print("\n[Plot] Learning curve …")
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=tscv, scoring=scorer, n_jobs=-1
    )

    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std  = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, tr_mean, "o-", color=PALETTE[0], label="Train R²")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color=PALETTE[0])
    ax.plot(train_sizes, va_mean, "s-", color=PALETTE[2], label="Validation R²")
    ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std, alpha=0.15, color=PALETTE[2])
    ax.set_title("Learning Curve – Tuned XGBoost", fontweight="bold")
    ax.set_xlabel("Training Samples"); ax.set_ylabel("R² Score")
    ax.legend()
    savefig("learning_curve")

    # ── Save best model & results ─────────────────────────────────────────────
    save_model(best_model, "best_model")

    results = {
        "best_params"         : {k: (int(v) if isinstance(v, np.integer) else
                                      float(v) if isinstance(v, np.floating) else v)
                                  for k, v in best_params.items()},
        "best_cv_r2_random"   : float(rscv.best_score_),
        "best_cv_r2_grid"     : float(gscv.best_score_),
        "test_metrics"        : final_metrics,
    }
    path = os.path.join(REPORTS_DIR, "tuning_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Tuning results saved → {path}")

    print(f"\n  🏆 Tuned model test R²={final_metrics['R2']:.4f}  RMSE={final_metrics['RMSE']:,.0f}")
    return best_model, final_metrics


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
