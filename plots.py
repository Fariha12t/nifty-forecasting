"""All figures. Matplotlib only, saved to PNG."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "font.size": 10})


def _dates(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_overview(close: pd.Series, ret: pd.Series, path):
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(close.index, close.values, lw=0.8)
    ax[0].set_title("NIFTY 50 closing price")
    ax[0].set_ylabel("Index level")
    ax[1].plot(ret.index, ret.values, lw=0.5)
    ax[1].set_title("Daily log returns")
    ax[1].set_ylabel("Return (%)")
    ax[2].plot(ret.index, ret.rolling(21).std() * np.sqrt(252), lw=0.8, color="tab:red")
    ax[2].set_title("21-day rolling volatility (annualised)")
    ax[2].set_ylabel("Volatility (%)")
    ax[2].set_xlabel("Date")
    _dates(ax[2])
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
    ax.set_xlabel("Month")
    _dates(ax)
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
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative squared-error difference")
    _dates(ax)
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
    ax.set_title("One-step-ahead daily volatility forecasts vs |return|")
    ax.set_ylabel("Volatility (% per day)")
    ax.set_xlabel("Date")
    _dates(ax)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_var(df: pd.DataFrame, model: str, path, last=750, level: float = 0.01):
    col = f"q{level}"
    d = df[df["model"] == model].tail(last)
    viol = d["actual"] < d[col]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(d["date"], d["actual"], color="grey", lw=0.5, label="Daily return")
    ax.plot(d["date"], d[col], color="tab:red", lw=1, label=f"{int((1 - level) * 100)}% VaR forecast")
    ax.scatter(d["date"][viol], d["actual"][viol], color="k", s=14, zorder=3, label="Exceedance")
    ax.set_title(f"1-day {int((1 - level) * 100)}% VaR: {model} (last {len(d)} days, {int(viol.sum())} exceedances)")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Date")
    _dates(ax)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_return_distribution(ret: pd.Series, path):
    x = ret.dropna().to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(x, bins=80, density=True, alpha=0.6, label="Empirical")
    grid = np.linspace(x.min(), x.max(), 400)
    ax[0].plot(grid, stats.norm.pdf(grid, x.mean(), x.std(ddof=1)), "r", lw=1.2, label="Normal (same mean, sd)")
    nu, loc, sc = stats.t.fit(x)
    ax[0].plot(grid, stats.t.pdf(grid, nu, loc, sc), "g--", lw=1.2, label=f"Student-t fit (nu={nu:.1f})")
    ax[0].set_yscale("log")
    ax[0].set_ylim(bottom=1e-4)
    ax[0].set_title("Daily log return distribution (log density)")
    ax[0].set_xlabel("Daily return (%)")
    ax[0].set_ylabel("Density")
    ax[0].legend(fontsize=8)
    stats.probplot(x, dist="norm", plot=ax[1])
    ax[1].set_title("Normal Q-Q plot of daily returns")
    ax[1].set_xlabel("Theoretical quantiles")
    ax[1].set_ylabel("Sample quantiles (%)")
    _save(fig, path)


def plot_monthly_returns(r_m: pd.Series, path):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(r_m.index, r_m.values, width=20, color=np.where(r_m.values >= 0, "tab:blue", "tab:red"))
    ax.set_title("Monthly log returns")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Month")
    _dates(ax)
    _save(fig, path)


def plot_monthly_acf_pacf(r_m: pd.Series, path, lags=24):
    x = r_m.dropna().to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    plot_acf(x, lags=lags, ax=ax[0], title="ACF: monthly returns")
    plot_pacf(x, lags=lags, ax=ax[1], title="PACF: monthly returns", method="ywm")
    for a in ax:
        a.set_xlabel("Lag (months)")
    _save(fig, path)


def plot_forecast_comparison(df: pd.DataFrame, path, last=48):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    base = df[df["model"] == "Historical mean"].tail(last)
    ax.plot(base["date"], base["actual"], "k.-", lw=0.8, ms=5, label="Actual", zorder=5)
    for m, g in df.groupby("model", sort=False):
        g = g.tail(last)
        ax.plot(g["date"], g["pred"], lw=1, label=m)
    ax.set_title(f"Actual vs one-step-ahead forecast monthly returns (last {last} months)")
    ax.set_ylabel("Monthly log return (%)")
    ax.set_xlabel("Month")
    _dates(ax)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_interval_comparison(df: pd.DataFrame, path, last=60):
    models = list(df["model"].unique())
    fig, ax = plt.subplots(len(models), 1, figsize=(10, 2.0 * len(models)), sharex=True)
    for a, m in zip(np.atleast_1d(ax), models):
        d = df[df["model"] == m].tail(last)
        a.fill_between(d["date"], d["lo"], d["hi"], alpha=0.25)
        a.plot(d["date"], d["pred"], lw=1)
        a.plot(d["date"], d["actual"], "k.", ms=3)
        a.set_ylabel("%")
        a.set_title(m, fontsize=9, loc="left")
    np.atleast_1d(ax)[-1].set_xlabel("Month")
    _dates(np.atleast_1d(ax)[-1])
    fig.suptitle("95% prediction intervals by model (dots = actual)")
    _save(fig, path)


def plot_volatility_clustering(ret: pd.Series, path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(ret.index, ret.values, lw=0.5)
    ax[0].set_title("Daily log returns: calm and turbulent periods cluster")
    ax[0].set_ylabel("Return (%)")
    ax[1].plot(ret.index, ret.abs().rolling(21).mean(), lw=0.9, color="tab:red")
    ax[1].set_title("21-day moving average of |return|")
    ax[1].set_ylabel("Mean |return| (%)")
    ax[1].set_xlabel("Date")
    _dates(ax[1])
    _save(fig, path)


def plot_conditional_vol(ret: pd.Series, results: dict, path):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(ret.index, ret.abs(), color="lightgrey", lw=0.5, label="|return|")
    for name, res in results.items():
        ax.plot(res.conditional_volatility.index, res.conditional_volatility, lw=0.8, label=name)
    ax.set_title("GARCH-family conditional volatility (full-sample fit, in-sample)")
    ax.set_ylabel("Volatility (% per day)")
    ax.set_xlabel("Date")
    _dates(ax)
    ax.legend(fontsize=8)
    _save(fig, path)
