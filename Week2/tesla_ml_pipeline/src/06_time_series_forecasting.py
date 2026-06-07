"""
06_time_series_forecasting.py
─────────────────────────────────────────────────────────────────
Stage 6 – Time Series Forecasting

Models
──────
1. SARIMA  – statsmodels auto-parameter selection via AIC grid search
2. Prophet – Facebook/Meta Prophet with yearly + monthly seasonality

Both models forecast global monthly deliveries 12 months ahead
(Jan 2026 – Dec 2026).

Outputs
───────
• outputs/ts_global_trend.png          – Historical global monthly deliveries
• outputs/ts_decomposition.png         – Seasonal decomposition (trend/seasonal/resid)
• outputs/ts_acf_pacf.png              – ACF & PACF plots
• outputs/forecast_sarima.png          – SARIMA 12-month forecast
• outputs/forecast_prophet.png         – Prophet forecast with uncertainty bands
• outputs/forecast_comparison.png      – SARIMA vs Prophet side by side
• reports/forecast_values.csv          – Numeric forecast table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal            import seasonal_decompose
from statsmodels.graphics.tsaplots       import plot_acf, plot_pacf
from itertools                           import product

from utils import DATA_RAW, OUTPUTS_DIR, REPORTS_DIR, savefig, section, PALETTE


# ─────────────────────────────────────────────────────────────────────────────
def _build_global_ts():
    """Aggregate all regions/models into a single monthly global series."""
    df = pd.read_csv(DATA_RAW)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    ts = (df.groupby("Date")["Estimated_Deliveries"]
            .sum()
            .sort_index()
            .asfreq("MS"))           # Month Start frequency
    ts.name = "Global_Deliveries"
    return ts


# ─────────────────────────────────────────────────────────────────────────────
def run():
    section("STAGE 6 – TIME SERIES FORECASTING")

    ts = _build_global_ts()
    print(f"\n  Time series: {ts.index[0].date()} → {ts.index[-1].date()}  |  {len(ts)} observations")
    print(f"  Range: {ts.min():,} – {ts.max():,}  |  Mean: {ts.mean():,.0f}")

    HORIZON = 12   # forecast periods (months)

    # ── Historical trend plot ────────────────────────────────────────────────
    print("\n[1] Global monthly deliveries plot …")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(ts.index, ts / 1e6, color=PALETTE[0], linewidth=1.5)
    ax.fill_between(ts.index, 0, ts / 1e6, alpha=0.12, color=PALETTE[0])
    ax.set_title("Global Monthly Estimated Deliveries (2015–2025)", fontweight="bold")
    ax.set_ylabel("Deliveries (millions)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    savefig("ts_global_trend")

    # ── Seasonal decomposition ───────────────────────────────────────────────
    print("[2] Seasonal decomposition …")
    decomp = seasonal_decompose(ts, model="additive", period=12, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for ax, comp, label, col in zip(
        axes,
        [ts, decomp.trend, decomp.seasonal, decomp.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
        PALETTE[:4]
    ):
        ax.plot(comp.index, comp / 1e6, color=col, linewidth=1.2)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="x", linewidth=0.4, alpha=0.5)

    axes[0].set_title("Seasonal Decomposition – Global Monthly Deliveries", fontweight="bold")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    savefig("ts_decomposition")

    # ── ACF / PACF ───────────────────────────────────────────────────────────
    print("[3] ACF / PACF …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf( ts, lags=36, ax=axes[0], color=PALETTE[0])
    plot_pacf(ts, lags=36, ax=axes[1], color=PALETTE[2], method="ols")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    savefig("ts_acf_pacf")

    # ═══════════════════════════════════════════════════════════════════════
    #  MODEL 1 – SARIMA
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[SARIMA] Grid search over (p,d,q)(P,D,Q,12) …")

    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal = (1, 1, 1, 12)

    p_vals = [0, 1, 2]; d_vals = [1]; q_vals = [0, 1, 2]
    P_vals = [0, 1];    D_vals = [1]; Q_vals = [0, 1]

    for (p, d, q), (P, D, Q) in product(
        product(p_vals, d_vals, q_vals),
        product(P_vals, D_vals, Q_vals)
    ):
        try:
            mdl = SARIMAX(ts,
                          order=(p, d, q),
                          seasonal_order=(P, D, Q, 12),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            res = mdl.fit(disp=False)
            if res.aic < best_aic:
                best_aic   = res.aic
                best_order = (p, d, q)
                best_seasonal = (P, D, Q, 12)
        except Exception:
            continue

    print(f"  Best SARIMA order : {best_order}  seasonal: {best_seasonal}  AIC={best_aic:.1f}")

    sarima_model = SARIMAX(ts,
                           order=best_order,
                           seasonal_order=best_seasonal,
                           enforce_stationarity=False,
                           enforce_invertibility=False).fit(disp=False)

    # In-sample residuals
    sarima_resid = sarima_model.resid
    resid_rmse = np.sqrt((sarima_resid ** 2).mean())
    print(f"  In-sample RMSE: {resid_rmse:,.0f}")

    # Forecast
    forecast_sarima = sarima_model.get_forecast(steps=HORIZON)
    fc_mean_s = forecast_sarima.predicted_mean
    fc_ci_s   = forecast_sarima.conf_int(alpha=0.20)   # 80% CI

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ts.index[-48:], ts.iloc[-48:] / 1e6, color=PALETTE[1], linewidth=1.5, label="Historical")
    ax.plot(fc_mean_s.index, fc_mean_s / 1e6, color=PALETTE[0], linewidth=2, label="SARIMA Forecast")
    ax.fill_between(fc_mean_s.index,
                    fc_ci_s.iloc[:, 0] / 1e6,
                    fc_ci_s.iloc[:, 1] / 1e6,
                    alpha=0.25, color=PALETTE[0], label="80% CI")
    ax.set_title(f"SARIMA{best_order}×{best_seasonal} – 12-Month Forecast", fontweight="bold")
    ax.set_ylabel("Deliveries (millions)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    savefig("forecast_sarima")

    # ═══════════════════════════════════════════════════════════════════════
    #  MODEL 2 – Prophet
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Prophet] Fitting …")
    try:
        from prophet import Prophet

        prophet_df = ts.reset_index()
        prophet_df.columns = ["ds", "y"]

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.80,
            changepoint_prior_scale=0.15,
        )
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        m.fit(prophet_df)

        future   = m.make_future_dataframe(periods=HORIZON, freq="MS")
        forecast  = m.predict(future)
        fc_prophet= forecast[forecast["ds"] > ts.index[-1]][["ds", "yhat", "yhat_lower", "yhat_upper"]]

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(prophet_df["ds"].iloc[-48:], prophet_df["y"].iloc[-48:] / 1e6,
                color=PALETTE[1], linewidth=1.5, label="Historical")
        ax.plot(fc_prophet["ds"], fc_prophet["yhat"] / 1e6,
                color=PALETTE[2], linewidth=2, label="Prophet Forecast")
        ax.fill_between(fc_prophet["ds"],
                        fc_prophet["yhat_lower"] / 1e6,
                        fc_prophet["yhat_upper"] / 1e6,
                        alpha=0.25, color=PALETTE[2], label="80% CI")
        ax.set_title("Prophet – 12-Month Forecast", fontweight="bold")
        ax.set_ylabel("Deliveries (millions)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=30)
        savefig("forecast_prophet")

        prophet_ok = True

    except ImportError:
        print("  ⚠  Prophet not installed – skipping. Run: pip install prophet")
        fc_prophet = None
        prophet_ok = False

    # ── Side-by-side comparison ──────────────────────────────────────────────
    if prophet_ok:
        print("\n[Plot] Forecast comparison …")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle("12-Month Forecast Comparison", fontsize=13, fontweight="bold")

        for ax, (label, fc_mean, fc_ci, color) in zip(
            axes,
            [
                ("SARIMA", fc_mean_s / 1e6, fc_ci_s / 1e6, PALETTE[0]),
                ("Prophet", fc_prophet["yhat"] / 1e6,
                 fc_prophet[["yhat_lower", "yhat_upper"]] / 1e6, PALETTE[2]),
            ]
        ):
            ax.plot(ts.index[-36:], ts.iloc[-36:] / 1e6, color=PALETTE[1],
                    linewidth=1.4, label="Historical")
            if label == "SARIMA":
                ax.plot(fc_mean.index, fc_mean, color=color, linewidth=2, label=label)
                ax.fill_between(fc_mean.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                                alpha=0.2, color=color)
            else:
                ax.plot(fc_prophet["ds"], fc_mean, color=color, linewidth=2, label=label)
                ax.fill_between(fc_prophet["ds"],
                                fc_ci["yhat_lower"], fc_ci["yhat_upper"],
                                alpha=0.2, color=color)
            ax.set_title(label, fontweight="bold")
            ax.set_ylabel("Deliveries (millions)")
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

        savefig("forecast_comparison")

    # ── Save numeric forecast table ──────────────────────────────────────────
    print("\n[CSV] Saving forecast values …")
    sarima_table = pd.DataFrame({
        "Date"            : fc_mean_s.index,
        "SARIMA_forecast" : fc_mean_s.values,
        "SARIMA_lower_80" : fc_ci_s.iloc[:, 0].values,
        "SARIMA_upper_80" : fc_ci_s.iloc[:, 1].values,
    })

    if prophet_ok:
        sarima_table["Prophet_forecast"] = fc_prophet["yhat"].values
        sarima_table["Prophet_lower_80"] = fc_prophet["yhat_lower"].values
        sarima_table["Prophet_upper_80"] = fc_prophet["yhat_upper"].values

    out_path = os.path.join(REPORTS_DIR, "forecast_values.csv")
    sarima_table.to_csv(out_path, index=False)
    print(f"  ✓ Forecast table saved → {out_path}")
    print("\n" + sarima_table.to_string(index=False))

    print("\n  ✓ Time series forecasting complete.")
    return sarima_table


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
