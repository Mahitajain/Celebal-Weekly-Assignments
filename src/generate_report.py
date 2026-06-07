"""
generate_report.py
──────────────────
Reads model_metrics.json, tuning_results.json, forecast_values.csv
and writes reports/pipeline_report.md
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
from datetime import datetime
from utils import REPORTS_DIR, section


def run():
    section("REPORT GENERATION")

    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines += [
        f"# Tesla EV ML Pipeline – Summary Report",
        f"*Generated: {ts}*\n",
        "---\n",
        "## 1. Dataset Overview\n",
        "| Attribute | Value |",
        "|-----------|-------|",
        "| Source | Kaggle – Tesla EV Deliveries & Production 2015–2025 |",
        "| Rows | 2,640 |",
        "| Columns | 12 |",
        "| Target | `Estimated_Deliveries` |",
        "| Regions | Europe, Asia, North America, Middle East |",
        "| Models | Model S, Model X, Model 3, Model Y, Cybertruck |\n",
    ]

    # Stage metrics
    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        lines += [
            "## 2. Regression Model Comparison (Test Set)\n",
            "| Model | MAE | RMSE | R² | MAPE (%) |",
            "|-------|-----|------|----|----------|",
        ]
        for m in sorted(metrics, key=lambda x: x["R2"], reverse=True):
            lines.append(
                f"| {m['label']} | {m['MAE']:,.0f} | {m['RMSE']:,.0f} "
                f"| {m['R2']:.4f} | {m['MAPE']:.2f} |"
            )
        best = max(metrics, key=lambda x: x["R2"])
        lines += [f"\n**Best model: {best['label']}** – R²={best['R2']:.4f}, RMSE={best['RMSE']:,.0f}\n"]

    # Tuning results
    tuning_path = os.path.join(REPORTS_DIR, "tuning_results.json")
    if os.path.exists(tuning_path):
        with open(tuning_path) as f:
            tr = json.load(f)

        lines += [
            "## 3. Hyperparameter Tuning (XGBoost)\n",
            f"- RandomizedSearch best CV R²: **{tr['best_cv_r2_random']:.4f}**",
            f"- GridSearch best CV R²:       **{tr['best_cv_r2_grid']:.4f}**",
            f"- Test R²:                     **{tr['test_metrics']['R2']:.4f}**",
            f"- Test RMSE:                   **{tr['test_metrics']['RMSE']:,.0f}**\n",
            "**Best parameters:**\n",
            "```json",
            json.dumps(tr["best_params"], indent=2),
            "```\n",
        ]

    # Forecast table
    fc_path = os.path.join(REPORTS_DIR, "forecast_values.csv")
    if os.path.exists(fc_path):
        fc = pd.read_csv(fc_path)
        fc["Date"] = pd.to_datetime(fc["Date"]).dt.strftime("%b %Y")
        lines += [
            "## 4. Time Series Forecast (Jan – Dec 2026)\n",
            fc.to_markdown(index=False),
            "\n",
        ]

    # Outputs
    lines += [
        "## 5. Generated Artefacts\n",
        "### Plots (`outputs/`)",
        "| File | Description |",
        "|------|-------------|",
        "| `eda_target_distribution.png` | Target variable distribution |",
        "| `eda_yearly_deliveries.png` | Annual delivery trend |",
        "| `eda_deliveries_by_region.png` | Stacked region breakdown |",
        "| `eda_deliveries_by_model.png` | Stacked model breakdown |",
        "| `eda_monthly_seasonality.png` | Heatmap: year × month |",
        "| `eda_price_vs_deliveries.png` | Price vs delivery scatter |",
        "| `eda_correlation_heatmap.png` | Pearson correlation matrix |",
        "| `eda_numeric_distributions.png` | KDE plots for all features |",
        "| `eda_production_vs_delivery.png` | Production vs delivery trend |",
        "| `eda_co2_trend.png` | CO₂ savings over time |",
        "| `eda_feature_importance_mi.png` | Mutual information ranking |",
        "| `model_comparison.png` | R² & RMSE comparison |",
        "| `feature_importance.png` | Random Forest importances |",
        "| `tuning_cv_scores.png` | RandomizedSearch CV scores |",
        "| `learning_curve.png` | Bias–variance learning curve |",
        "| `ts_global_trend.png` | Global monthly TS |",
        "| `ts_decomposition.png` | Trend/seasonal decomposition |",
        "| `ts_acf_pacf.png` | ACF & PACF plots |",
        "| `forecast_sarima.png` | SARIMA 12-month forecast |",
        "| `forecast_prophet.png` | Prophet 12-month forecast |",
        "| `forecast_comparison.png` | SARIMA vs Prophet |",
        "\n### Models (`models/`)",
        "All 6 regression models are saved as `.pkl` files plus `best_model.pkl` (tuned XGBoost).\n",
    ]

    report = "\n".join(lines)
    out_path = os.path.join(REPORTS_DIR, "pipeline_report.md")
    with open(out_path, "w") as f:
        f.write(report)

    print(f"  ✓ Report saved → {out_path}")
    return out_path


if __name__ == "__main__":
    run()
