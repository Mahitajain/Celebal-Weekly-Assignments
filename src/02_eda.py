"""
02_eda.py
─────────────────────────────────────────────────────────────────
Stage 2 – Exploratory Data Analysis

Plots generated
───────────────
1.  eda_target_distribution          (done in preprocessing)
2.  eda_yearly_deliveries            Global deliveries by year
3.  eda_deliveries_by_region         Region breakdown stacked bar
4.  eda_deliveries_by_model          Model breakdown stacked bar
5.  eda_monthly_seasonality          Average deliveries by month (heatmap)
6.  eda_price_vs_deliveries          Scatter: Avg_Price_USD vs deliveries
7.  eda_correlation_heatmap          Pearson correlations of numeric cols
8.  eda_numeric_distributions        KDE grid for all numeric features
9.  eda_production_vs_delivery       Production vs delivery trend line
10. eda_co2_trend                    CO₂ saved over years
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from utils import DATA_PROC, savefig, section, PALETTE

sns.set_theme(style="whitegrid", font_scale=1.05)


# ─────────────────────────────────────────────────────────────────────────────
def _load():
    df = pd.read_csv(DATA_PROC, parse_dates=["Date"])
    # Recover original string cols from one-hot columns
    region_cols = [c for c in df.columns if c.startswith("R_")]
    model_cols  = [c for c in df.columns if c.startswith("M_")]

    if region_cols:
        df["Region"] = (
            pd.from_dummies(df[region_cols])
              .apply(lambda r: r.idxmax(), axis=1)
              .str.replace("R_", "", regex=False)
        )
    if model_cols:
        df["Model"] = (
            pd.from_dummies(df[model_cols])
              .apply(lambda r: r.idxmax(), axis=1)
              .str.replace("M_", "", regex=False)
        )
    return df


def run():
    section("STAGE 2 – EDA")
    df = _load()

    numeric_cols = ["Estimated_Deliveries", "Production_Units", "Avg_Price_USD",
                    "Battery_Capacity_kWh", "Range_km", "CO2_Saved_tons", "Charging_Stations"]

    # ── 1. Yearly global deliveries ──────────────────────────────────────────
    print("\n[1] Yearly deliveries trend …")
    yearly = df.groupby("Year")["Estimated_Deliveries"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(yearly["Year"], yearly["Estimated_Deliveries"] / 1e6,
           color=PALETTE[0], edgecolor="white", linewidth=0.6)
    ax.plot(yearly["Year"], yearly["Estimated_Deliveries"] / 1e6,
            color=PALETTE[1], marker="o", linewidth=2)
    ax.set_title("Global Estimated Deliveries by Year", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Deliveries (millions)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    savefig("eda_yearly_deliveries")

    # ── 2. Deliveries by Region ──────────────────────────────────────────────
    print("[2] Deliveries by region …")
    reg = df.groupby(["Year", "Region"])["Estimated_Deliveries"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    reg.div(1e6).plot(kind="bar", stacked=True, ax=ax,
                      color=PALETTE[:len(reg.columns)], edgecolor="white", linewidth=0.4)
    ax.set_title("Deliveries by Region (Stacked)", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Deliveries (millions)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=45)
    savefig("eda_deliveries_by_region")

    # ── 3. Deliveries by Model ───────────────────────────────────────────────
    print("[3] Deliveries by model …")
    mod = df.groupby(["Year", "Model"])["Estimated_Deliveries"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    mod.div(1e6).plot(kind="bar", stacked=True, ax=ax,
                      color=PALETTE[:len(mod.columns)], edgecolor="white", linewidth=0.4)
    ax.set_title("Deliveries by Model (Stacked)", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Deliveries (millions)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=45)
    savefig("eda_deliveries_by_model")

    # ── 4. Monthly seasonality heatmap ───────────────────────────────────────
    print("[4] Monthly seasonality heatmap …")
    pivot = df.pivot_table(values="Estimated_Deliveries",
                           index="Year", columns="Month", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(pivot / 1e3, ax=ax, cmap="YlOrRd", linewidths=0.3,
                annot=True, fmt=".0f", annot_kws={"size": 7},
                cbar_kws={"label": "Avg Deliveries (thousands)"})
    ax.set_title("Monthly Avg Deliveries Heatmap (Year × Month)", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Year")
    savefig("eda_monthly_seasonality")

    # ── 5. Price vs Deliveries scatter ───────────────────────────────────────
    print("[5] Price vs Deliveries …")
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(df["Avg_Price_USD"], df["Estimated_Deliveries"],
                    c=df["Year"], cmap="plasma", alpha=0.6, s=18, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Year")
    ax.set_title("Avg Price vs Estimated Deliveries", fontweight="bold")
    ax.set_xlabel("Average Price (USD)")
    ax.set_ylabel("Estimated Deliveries")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    savefig("eda_price_vs_deliveries")

    # ── 6. Correlation heatmap ───────────────────────────────────────────────
    print("[6] Correlation heatmap …")
    corr = df[numeric_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Matrix", fontweight="bold")
    savefig("eda_correlation_heatmap")

    # ── 7. KDE distributions ─────────────────────────────────────────────────
    print("[7] Numeric KDE distributions …")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=35, color=PALETTE[i % len(PALETTE)],
                     edgecolor="white", linewidth=0.4, density=True, alpha=0.75)
        df[col].plot.kde(ax=axes[i], color="black", linewidth=1.5)
        axes[i].set_title(col, fontsize=9, fontweight="bold")
        axes[i].set_xlabel("")
    axes[-1].set_visible(False)
    fig.suptitle("Numeric Feature Distributions (KDE)", fontsize=13, fontweight="bold")
    savefig("eda_numeric_distributions")

    # ── 8. Production vs Delivery trend ─────────────────────────────────────
    print("[8] Production vs Delivery trend …")
    trend = df.groupby("Year")[["Estimated_Deliveries", "Production_Units"]].sum() / 1e6

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trend.index, trend["Estimated_Deliveries"], marker="o",
            color=PALETTE[0], linewidth=2, label="Deliveries")
    ax.plot(trend.index, trend["Production_Units"], marker="s",
            color=PALETTE[2], linewidth=2, linestyle="--", label="Production")
    ax.fill_between(trend.index, trend["Estimated_Deliveries"],
                    trend["Production_Units"], alpha=0.12, color=PALETTE[2])
    ax.set_title("Production vs Delivery – Annual Totals", fontweight="bold")
    ax.set_ylabel("Units (millions)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend()
    savefig("eda_production_vs_delivery")

    # ── 9. CO₂ trend ─────────────────────────────────────────────────────────
    print("[9] CO₂ saved trend …")
    co2 = df.groupby("Year")["CO2_Saved_tons"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(co2["Year"], co2["CO2_Saved_tons"] / 1e6,
                    alpha=0.35, color="#70AD47")
    ax.plot(co2["Year"], co2["CO2_Saved_tons"] / 1e6,
            color="#70AD47", marker="o", linewidth=2)
    ax.set_title("Cumulative CO₂ Saved by Year (megatons)", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("CO₂ Saved (Mt)")
    savefig("eda_co2_trend")

    print("\n  ✓ All EDA plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
