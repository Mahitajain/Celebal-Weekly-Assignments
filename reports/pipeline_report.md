# Tesla EV ML Pipeline – Summary Report
*Generated: 2026-06-07 07:48*

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| Source | Kaggle – Tesla EV Deliveries & Production 2015–2025 |
| Rows | 2,640 |
| Columns | 12 |
| Target | `Estimated_Deliveries` |
| Regions | Europe, Asia, North America, Middle East |
| Models | Model S, Model X, Model 3, Model Y, Cybertruck |

## 2. Regression Model Comparison (Test Set)

| Model | MAE | RMSE | R² | MAPE (%) |
|-------|-----|------|----|----------|
| Linear Regression | 0 | 0 | 1.0000 | 0.00 |
| Ridge | 36 | 45 | 0.9999 | 0.45 |
| Gradient Boosting | 48 | 67 | 0.9997 | 0.57 |
| Random Forest | 77 | 115 | 0.9990 | 0.96 |
| Lasso | 98 | 118 | 0.9990 | 1.22 |
| XGBoost | 139 | 190 | 0.9973 | 1.60 |

**Best model: Linear Regression** – R²=1.0000, RMSE=0

## 3. Hyperparameter Tuning (XGBoost)

- RandomizedSearch best CV R²: **0.9972**
- GridSearch best CV R²:       **0.9973**
- Test R²:                     **0.9994**
- Test RMSE:                   **89**

**Best parameters:**

```json
{
  "subsample": 1.0,
  "reg_lambda": 1.5,
  "reg_alpha": 0,
  "min_child_weight": 3,
  "gamma": 0,
  "colsample_bytree": 1.0,
  "learning_rate": 0.09000000000000001,
  "max_depth": 4,
  "n_estimators": 200
}
```

## 4. Time Series Forecast (Jan – Dec 2026)

| Date     |   SARIMA_forecast |   SARIMA_lower_80 |   SARIMA_upper_80 |   Prophet_forecast |   Prophet_lower_80 |   Prophet_upper_80 |
|:---------|------------------:|------------------:|------------------:|-------------------:|-------------------:|-------------------:|
| Jan 2026 |            201663 |            170669 |            232656 |             194659 |             176278 |             213039 |
| Feb 2026 |            188044 |            156122 |            219967 |             191296 |             172126 |             209593 |
| Mar 2026 |            201932 |            170008 |            233856 |             177910 |             158482 |             196125 |
| Apr 2026 |            200960 |            168810 |            233111 |             189695 |             172120 |             208671 |
| May 2026 |            188911 |            155788 |            222034 |             177506 |             158967 |             195849 |
| Jun 2026 |            191617 |            157455 |            225779 |             184129 |             165565 |             202905 |
| Jul 2026 |            197944 |            163030 |            232859 |             179767 |             161895 |             197341 |
| Aug 2026 |            208816 |            173279 |            244354 |             197634 |             179086 |             216666 |
| Sep 2026 |            193410 |            157238 |            229582 |             190076 |             171673 |             209781 |
| Oct 2026 |            188702 |            151862 |            225541 |             192797 |             172505 |             210921 |
| Nov 2026 |            197667 |            160162 |            235173 |             193630 |             174886 |             211339 |
| Dec 2026 |            201732 |            163581 |            239883 |             191875 |             173079 |             210698 |


## 5. Generated Artefacts

### Plots (`outputs/`)
| File | Description |
|------|-------------|
| `eda_target_distribution.png` | Target variable distribution |
| `eda_yearly_deliveries.png` | Annual delivery trend |
| `eda_deliveries_by_region.png` | Stacked region breakdown |
| `eda_deliveries_by_model.png` | Stacked model breakdown |
| `eda_monthly_seasonality.png` | Heatmap: year × month |
| `eda_price_vs_deliveries.png` | Price vs delivery scatter |
| `eda_correlation_heatmap.png` | Pearson correlation matrix |
| `eda_numeric_distributions.png` | KDE plots for all features |
| `eda_production_vs_delivery.png` | Production vs delivery trend |
| `eda_co2_trend.png` | CO₂ savings over time |
| `eda_feature_importance_mi.png` | Mutual information ranking |
| `model_comparison.png` | R² & RMSE comparison |
| `feature_importance.png` | Random Forest importances |
| `tuning_cv_scores.png` | RandomizedSearch CV scores |
| `learning_curve.png` | Bias–variance learning curve |
| `ts_global_trend.png` | Global monthly TS |
| `ts_decomposition.png` | Trend/seasonal decomposition |
| `ts_acf_pacf.png` | ACF & PACF plots |
| `forecast_sarima.png` | SARIMA 12-month forecast |
| `forecast_prophet.png` | Prophet 12-month forecast |
| `forecast_comparison.png` | SARIMA vs Prophet |

### Models (`models/`)
All 6 regression models are saved as `.pkl` files plus `best_model.pkl` (tuned XGBoost).
