"""
01_preprocessing.py
────────────────────
Stage 1 – Data Loading, Cleaning & Encoding

Steps
-----
1. Load raw CSV
2. Audit nulls, duplicates, dtypes
3. Cast / coerce types
4. Create a proper DatetimeIndex
5. Encode categoricals (Label + One-Hot)
6. Remove statistical outliers (IQR on target)
7. Save cleaned dataset to data/processed_data.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import DATA_RAW, DATA_PROC, OUTPUTS_DIR, savefig, section


# ─────────────────────────────────────────────────────────────────────────────
def run():
    section("STAGE 1 – PREPROCESSING")

    # 1. Load ─────────────────────────────────────────────────────────────────
    print("\n[1] Loading raw data …")
    df = pd.read_csv(DATA_RAW)
    print(f"  Raw shape : {df.shape}")
    print(df.dtypes.to_string())

    # 2. Audit ────────────────────────────────────────────────────────────────
    print("\n[2] Null audit …")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0] if null_counts.any() else "  No nulls found ✓")

    print("\n  Duplicate rows:", df.duplicated().sum())

    # 3. Type casting ─────────────────────────────────────────────────────────
    print("\n[3] Casting types …")
    int_cols   = ["Year", "Month", "Estimated_Deliveries", "Production_Units",
                  "Battery_Capacity_kWh", "Range_km", "Charging_Stations"]
    float_cols = ["Avg_Price_USD", "CO2_Saved_tons"]

    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(subset=["Estimated_Deliveries"], inplace=True)
    df = df.astype({c: int for c in int_cols})   # back to plain int after dropna

    # 4. DateTime ─────────────────────────────────────────────────────────────
    print("\n[4] Building Date column …")
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 5. Categorical Encoding ─────────────────────────────────────────────────
    print("\n[5] Encoding categoricals …")

    # Label encode (for tree models)
    from sklearn.preprocessing import LabelEncoder
    le_region = LabelEncoder()
    le_model  = LabelEncoder()
    le_source = LabelEncoder()

    df["Region_enc"]  = le_region.fit_transform(df["Region"])
    df["Model_enc"]   = le_model.fit_transform(df["Model"])
    df["Source_enc"]  = le_source.fit_transform(df["Source_Type"])

    # One-hot encode (for linear models)
    df = pd.get_dummies(df, columns=["Region", "Model", "Source_Type"],
                        prefix=["R", "M", "S"], drop_first=False)

    print(f"  Columns after encoding: {df.shape[1]}")

    # 6. Outlier removal on target ────────────────────────────────────────────
    print("\n[6] IQR outlier removal on Estimated_Deliveries …")
    Q1 = df["Estimated_Deliveries"].quantile(0.01)
    Q3 = df["Estimated_Deliveries"].quantile(0.99)
    IQR = Q3 - Q1
    before = len(df)
    df = df[(df["Estimated_Deliveries"] >= Q1 - 1.5 * IQR) &
            (df["Estimated_Deliveries"] <= Q3 + 1.5 * IQR)]
    print(f"  Removed {before - len(df)} outlier rows  |  Remaining: {len(df)}")

    # 7. Visualise target distribution ────────────────────────────────────────
    print("\n[7] Plotting target distribution …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Target Distribution – Estimated Deliveries", fontsize=13, fontweight="bold")

    axes[0].hist(df["Estimated_Deliveries"], bins=40, color="#E31937", edgecolor="white", linewidth=0.5)
    axes[0].set_title("Raw Distribution")
    axes[0].set_xlabel("Estimated Deliveries")

    axes[1].hist(np.log1p(df["Estimated_Deliveries"]), bins=40, color="#1B1B1B", edgecolor="white", linewidth=0.5)
    axes[1].set_title("Log-Transformed Distribution")
    axes[1].set_xlabel("log(1 + Estimated Deliveries)")

    savefig("eda_target_distribution")

    # 8. Save ─────────────────────────────────────────────────────────────────
    print("\n[8] Saving processed dataset …")
    df.to_csv(DATA_PROC, index=False)
    print(f"  ✓ Saved → {DATA_PROC}  |  Shape: {df.shape}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
