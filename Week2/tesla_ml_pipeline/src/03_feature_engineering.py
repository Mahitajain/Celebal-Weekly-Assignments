"""
03_feature_engineering.py
─────────────────────────────────────────────────────────────────
Stage 3 – Feature Engineering & Selection

Features created
────────────────
• Temporal          : quarter, week_of_year, is_Q4, year_progress
• Lag features      : deliveries / price lag 1, 3, 6 months
• Rolling stats     : 3-month & 6-month rolling mean / std
• Interaction terms : price × range, battery × range
• Log transforms    : log1p of skewed target & price
• VIF analysis      : flag / drop high-multicollinearity columns
• Mutual-info rank  : plot top 20 features
• Final feature set : saved to data/features.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import DATA_PROC, OUTPUTS_DIR, savefig, section, PALETTE

FEATURES_PATH = os.path.join(os.path.dirname(DATA_PROC), "features.csv")


# ─────────────────────────────────────────────────────────────────────────────
def run():
    section("STAGE 3 – FEATURE ENGINEERING")

    df = pd.read_csv(DATA_PROC, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Temporal features ──────────────────────────────────────────────────
    print("\n[1] Temporal features …")
    df["Quarter"]       = df["Month"].apply(lambda m: (m - 1) // 3 + 1)
    df["Week_of_year"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["Is_Q4"]         = (df["Quarter"] == 4).astype(int)
    df["Year_progress"] = (df["Month"] - 1) / 11   # 0.0 – 1.0

    # ── Lag features (sorted by Date, then grouped by Model+Region via enc) ─
    print("[2] Lag & rolling features …")

    # Use encoded columns as group proxy
    group_cols = ["Model_enc", "Region_enc"]
    df = df.sort_values(["Model_enc", "Region_enc", "Date"])

    for lag in [1, 3, 6]:
        df[f"Deliveries_lag{lag}"] = (
            df.groupby(group_cols)["Estimated_Deliveries"]
              .shift(lag)
        )
        df[f"Price_lag{lag}"] = (
            df.groupby(group_cols)["Avg_Price_USD"]
              .shift(lag)
        )

    for win in [3, 6]:
        df[f"Deliveries_roll{win}_mean"] = (
            df.groupby(group_cols)["Estimated_Deliveries"]
              .transform(lambda x: x.shift(1).rolling(win, min_periods=1).mean())
        )
        df[f"Deliveries_roll{win}_std"] = (
            df.groupby(group_cols)["Estimated_Deliveries"]
              .transform(lambda x: x.shift(1).rolling(win, min_periods=1).std().fillna(0))
        )

    # ── Interaction / ratio features ────────────────────────────────────────
    print("[3] Interaction & ratio features …")
    df["Price_x_Range"]    = df["Avg_Price_USD"] * df["Range_km"]
    df["Battery_x_Range"]  = df["Battery_Capacity_kWh"] * df["Range_km"]
    df["Efficiency"]       = df["Range_km"] / df["Battery_Capacity_kWh"]
    df["Prod_Delivery_gap"]= df["Production_Units"] - df["Estimated_Deliveries"]
    df["Station_per_Del"]  = df["Charging_Stations"] / (df["Estimated_Deliveries"] + 1)

    # ── Log transforms ──────────────────────────────────────────────────────
    print("[4] Log transforms …")
    df["log_Price"]       = np.log1p(df["Avg_Price_USD"])
    df["log_Deliveries"]  = np.log1p(df["Estimated_Deliveries"])    # log-target variant

    # ── Drop rows with NaN from lags ────────────────────────────────────────
    before = len(df)
    df.dropna(inplace=True)
    print(f"  Dropped {before - len(df)} rows with NaN after lag creation  |  Remaining: {len(df)}")

    # ── VIF analysis ────────────────────────────────────────────────────────
    print("\n[5] VIF analysis (top numeric features) …")

    vif_cols = [
        "Year", "Month", "Quarter", "Battery_Capacity_kWh", "Range_km",
        "Avg_Price_USD", "CO2_Saved_tons", "Charging_Stations",
        "Price_x_Range", "Battery_x_Range", "Efficiency",
        "Deliveries_lag1", "Deliveries_lag3", "Deliveries_roll3_mean",
    ]
    vif_cols = [c for c in vif_cols if c in df.columns]
    X_vif = df[vif_cols].astype(float)

    vif_df = pd.DataFrame({
        "Feature": vif_cols,
        "VIF": [variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])]
    }).sort_values("VIF", ascending=False)

    print(vif_df.to_string(index=False))

    # Drop features with VIF > 30 (extreme multicollinearity)
    high_vif = vif_df[vif_df["VIF"] > 30]["Feature"].tolist()
    print(f"\n  Dropping high-VIF features (>30): {high_vif}")
    df.drop(columns=[c for c in high_vif if c in df.columns], inplace=True, errors="ignore")

    # ── Mutual Information ranking ───────────────────────────────────────────
    print("\n[6] Mutual information ranking …")

    TARGET = "Estimated_Deliveries"
    exclude = {TARGET, "log_Deliveries", "Date"}
    ohe_cols = [c for c in df.columns if c.startswith(("R_", "M_", "S_"))]
    feat_cols = [c for c in df.columns
                 if c not in exclude and df[c].dtype in [np.float64, np.int64, int, float]
                 and c not in ohe_cols]

    X_mi = df[feat_cols].fillna(0)
    y_mi = df[TARGET]
    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
    mi_df = (pd.Series(mi_scores, index=feat_cols)
               .sort_values(ascending=False)
               .head(20))

    fig, ax = plt.subplots(figsize=(10, 6))
    mi_df[::-1].plot(kind="barh", ax=ax, color=PALETTE[0], edgecolor="white")
    ax.set_title("Top 20 Features – Mutual Information with Target", fontweight="bold")
    ax.set_xlabel("Mutual Information Score")
    savefig("eda_feature_importance_mi")

    # ── Save feature set ─────────────────────────────────────────────────────
    print(f"\n[7] Saving feature dataset …")
    df.to_csv(FEATURES_PATH, index=False)
    print(f"  ✓ Saved → {FEATURES_PATH}  |  Shape: {df.shape}")
    print(f"  Final columns ({len(df.columns)}): {df.columns.tolist()[:15]} …")

    return df, feat_cols + ohe_cols


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
