"""All figures. Matplotlib only, saved to PNG."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overview(close: pd.Series, ret: pd.Series, path):
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(close.index, close.values, lw=0.8)
    ax[0].set_title("NIFTY 50 close")
    ax[1].plot(ret.index, ret.values, lw=0.5)
    ax[1].set_title("Daily log returns (%)")
    ax[2].plot(ret.index, ret.rolling(21).std() * np.sqrt(252), lw=0.8, color="tab:red")
    ax[2].set_title("21-day rolling volatility (annualised %)")
    _save(fig, path)


def plot_acf_pacf(ret: pd.Series, path, lags=40):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    x = ret.dropna().to_numpy()
    plot_acf(x, lags=lags, ax=ax[0, 0], title="ACF: returns")
    plot_pacf(x, lags=lags, ax=ax[0, 1], title="PACF: returns", method="ywm")
    plot_acf(x**2, lags=lags, ax=ax[1, 0], title="ACF: squared returns (volatility clustering)")
    plot_acf(np.abs(x), lags=lags, ax=ax[1, 1], title="ACF: absolute returns")
    _save(fig, path)


def plot_residual_diagnostics(resid, title: str, path, lags=24):
    x = np.asarray(resid, dtype=float)
    x = x[~np.isnan(x)]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    ax[0, 0].plot(x, lw=0.6)
    ax[0, 0].axhline(0, color="k", lw=0.6)
    ax[0, 0].set_title("Residuals")
    ax[0, 1].hist(x, bins=40, density=True, alpha=0.7)
    grid = np.linspace(x.min(), x.max(), 200)
    ax[0, 1].plot(grid, stats.norm.pdf(grid, x.mean(), x.std()), "r", lw=1)
    ax[0, 1].set_title("Histogram vs normal")
    stats.probplot(x, dist="norm", plot=ax[1, 0])
    ax[1, 0].set_title("Normal Q-Q")
    plot_acf(x, lags=min(lags, len(x) // 2 - 1), ax=ax[1, 1], title="Residual ACF")
    fig.suptitle(title)
    _save(fig, path)


def plot_forecast_intervals(df: pd.DataFrame, model: str, path, last=60):
    d = df[df["model"] == model].tail(last)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(d["date"], d["lo"], d["hi"], alpha=0.25, label="95% prediction interval")
    ax.plot(d["date"], d["pred"], lw=1.2, label=f"{model} forecast")
    ax.plot(d["date"], d["actual"], "k.", ms=4, label="Actual")
    ax.set_title(f"One-step-ahead monthly return forecasts: {model} (last {len(d)} months)")
    ax.set_ylabel("Monthly log return (%)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_cum_sse_diff(df: pd.DataFrame, benchmark: str, path):
    """Cumulative squared-error difference vs the benchmark. Rising line = model beating the benchmark."""
    b = df[df["model"] == benchmark].set_index("date")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m, g in df.groupby("model"):
        if m == benchmark:
            continue
        g = g.set_index("date")
        diff = (b["actual"] - b["pred"]) ** 2 - (g["actual"] - g["pred"]) ** 2
        ax.plot(diff.index, diff.cumsum(), label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Cumulative SSE advantage over '{benchmark}' (above 0 = better)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_metric_bars(metrics: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    m = metrics.sort_values("RMSE")
    ax.barh(m["model"], m["RMSE"])
    ax.invert_yaxis()
    ax.set_xlabel("Out-of-sample RMSE (monthly return, %)")
    ax.set_title("Rolling-origin RMSE by model")
    lo = m["RMSE"].min() * 0.95
    ax.set_xlim(lo, m["RMSE"].max() * 1.02)
    _save(fig, path)


def plot_vol_forecasts(vol: pd.DataFrame, models: list[str], path, last=750):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    first = True
    for m in models:
        d = vol[vol["model"] == m].tail(last)
        if first:
            ax.plot(d["date"], np.abs(d["actual"]), color="lightgrey", lw=0.6, label="|return|")
            first = False
        ax.plot(d["date"], np.sqrt(d["sigma2"]), lw=1, label=m)
    ax.set_title("One-step-ahead daily volatility forecasts (% per day)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_var(df: pd.DataFrame, model: str, path, last=750):
    d = df[df["model"] == model].tail(last)
    viol = d["actual"] < d["q0.01"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(d["date"], d["actual"], color="grey", lw=0.5, label="Return")
    ax.plot(d["date"], d["q0.01"], color="tab:red", lw=1, label="1% VaR")
    ax.scatter(d["date"][viol], d["actual"][viol], color="k", s=14, zorder=3, label="Violation")
    ax.set_title(f"1-day 99% VaR: {model} (last {len(d)} days, {viol.sum()} violations)")
    ax.legend(fontsize=8)
    _save(fig, path)
