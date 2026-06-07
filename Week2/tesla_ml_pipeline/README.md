# Tesla EV Deliveries ML Pipeline (2015–2025)

An end-to-end machine learning pipeline on Tesla's global delivery and production data covering preprocessing, EDA, feature engineering, regression modeling, hyperparameter tuning, and time series forecasting.

## Dataset
- **Source**: [Kaggle – Tesla EV Deliveries & Production Data 2015–2025](https://www.kaggle.com/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025/data)
- **Records**: 2,640 rows × 12 columns
- **Granularity**: Monthly, by Region and Model

## Project Structure

```
tesla_ml_pipeline/
├── data/
│   └── raw/                        # Original CSV
├── src/
│   ├── 01_preprocessing.py         # Data cleaning & encoding
│   ├── 02_eda.py                   # Exploratory Data Analysis + plots
│   ├── 03_feature_engineering.py   # Feature creation & selection
│   ├── 04_regression_modeling.py   # Train/test regression models
│   ├── 05_hyperparameter_tuning.py # GridSearchCV / RandomizedSearchCV
│   ├── 06_time_series_forecasting.py # SARIMA + Prophet forecasting
│   └── utils.py                    # Shared helpers
├── models/                         # Saved model artifacts (.pkl)
├── outputs/                        # All generated plots & CSVs
├── reports/
│   └── pipeline_report.md          # Auto-generated summary report
├── run_pipeline.py                 # 🚀 Single entry point – runs everything
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd tesla_ml_pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python run_pipeline.py
```

All plots are saved to `outputs/`, trained models to `models/`, and a summary report is written to `reports/pipeline_report.md`.

## Pipeline Stages

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_preprocessing.py` | Null checks, type casting, label/one-hot encoding |
| 2 | `02_eda.py` | Distribution plots, correlation heatmap, trend charts |
| 3 | `03_feature_engineering.py` | Lag features, rolling stats, interaction terms, VIF |
| 4 | `04_regression_modeling.py` | LinearReg, Ridge, Lasso, RF, GBM, XGBoost |
| 5 | `05_hyperparameter_tuning.py` | RandomizedSearchCV on best model |
| 6 | `06_time_series_forecasting.py` | SARIMA + Prophet on global monthly deliveries |

## Models Evaluated

- Linear Regression
- Ridge & Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor *(tuned)*

## Key Outputs

- `outputs/eda_*.png` – EDA visualizations
- `outputs/model_comparison.png` – R² / RMSE bar charts
- `outputs/feature_importance.png` – Top predictors
- `outputs/forecast_sarima.png` – SARIMA 12-month forecast
- `outputs/forecast_prophet.png` – Prophet forecast with uncertainty bands
- `models/best_model.pkl` – Best tuned regressor
- `reports/pipeline_report.md` – Full metrics summary

## Target Variable
`Estimated_Deliveries` – monthly estimated EV deliveries per region/model.
