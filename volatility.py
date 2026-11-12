"""Daily volatility forecasting with GARCH-family models and risk baselines (rolling origin, 1-step ahead).

Parameters are re-estimated every `refit_every` days on data strictly before the forecast block; the
one-step variance forecast for each day in the block is then computed from the fixed parameters
using only past observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

SPECS = {
    "GARCH(1,1)-t": dict(vol="GARCH", p=1, o=0, q=1),
    "GJR-GARCH(1,1)-t": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1)-t": dict(vol="EGARCH", p=1, o=1, q=1),
}


def _model(y: pd.Series, spec: dict):
    return arch_model(y, mean="Constant", dist="t", rescale=False, **spec)


def garch_forecast_block(y: pd.Series, t: int, k: int, spec: dict):
    """Fit on y[:t]; return variance forecasts for positions t..t+k-1 plus mu and nu."""
    res = _model(y.iloc[:t], spec).fit(disp="off", show_warning=False)
    fixed = _model(y, spec).fix(res.params)
    fc = fixed.forecast(start=y.index[t - 1], horizon=1, reindex=False)
    var = fc.variance.iloc[:k, 0].to_numpy()
    return var, float(res.params["mu"]), float(res.params["nu"])


def rolling_garch(y: pd.Series, name: str, init: int, refit_every: int, max_origins: int | None) -> pd.DataFrame:
    spec = SPECS[name]
    n = len(y)
    start = max(init, n - max_origins) if max_origins else init
    out = []
    for t in range(start, n, refit_every):
        k = min(refit_every, n - t)
        try:
            var, mu, nu = garch_forecast_block(y, t, k, spec)
        except Exception:
            continue
        out.append(
            pd.DataFrame(
                {"model": name, "date": y.index[t : t + k], "actual": y.iloc[t : t + k].to_numpy(),
                 "sigma2": var, "mu": mu, "nu": nu}
            )
        )
    return pd.concat(out, ignore_index=True)


def ewma_variance(y: pd.Series, lam: float = 0.94, warm: int = 30) -> pd.Series:
    """RiskMetrics EWMA; value at date d is the forecast for d using returns through d-1."""
    r2 = y.to_numpy() ** 2
    s2 = np.empty(len(y))
    s2[0] = np.var(y.to_numpy()[:warm])
    for t in range(1, len(y)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r2[t - 1]
    return pd.Series(s2, index=y.index)


def rolling_window_variance(y: pd.Series, window: int = 250) -> pd.Series:
    """Variance of the previous `window` returns (excludes the current day)."""
    return (y.rolling(window).var()).shift(1)


def baseline_forecasts(y: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for name, s2 in (("EWMA (0.94)", ewma_variance(y)), ("Rolling 250d", rolling_window_variance(y))):
        sub = s2.reindex(dates)
        frames.append(
            pd.DataFrame(
                {"model": name, "date": dates, "actual": y.reindex(dates).to_numpy(),
                 "sigma2": sub.to_numpy(), "mu": 0.0, "nu": np.nan}
            )
        )
    return pd.concat(frames, ignore_index=True)


def standardized_quantile(alpha, nu):
    """Quantile of a unit-variance standardised t (or normal when nu is NaN)."""
    alpha = np.asarray(alpha, dtype=float)
    nu = np.asarray(nu, dtype=float)
    q_t = stats.t.ppf(alpha, np.where(np.isnan(nu), 100.0, nu)) * np.sqrt(
        (np.where(np.isnan(nu), 100.0, nu) - 2) / np.where(np.isnan(nu), 100.0, nu)
    )
    return np.where(np.isnan(nu), stats.norm.ppf(alpha), q_t)


def add_quantiles(df: pd.DataFrame, levels=(0.01, 0.05, 0.025, 0.975)) -> pd.DataFrame:
    df = df.copy()
    sigma = np.sqrt(df["sigma2"].to_numpy())
    for a in levels:
        df[f"q{a}"] = df["mu"].to_numpy() + sigma * standardized_quantile(a, df["nu"].to_numpy())
    return df


def fit_full_sample(y: pd.Series) -> pd.DataFrame:
    """Full-sample fits for the parameter table and standardised residual diagnostics."""
    rows, results = [], {}
    for name, spec in SPECS.items():
        res = _model(y, spec).fit(disp="off", show_warning=False)
        results[name] = res
        row = {"model": name, "loglik": res.loglikelihood, "AIC": res.aic, "BIC": res.bic}
        row.update({k: float(v) for k, v in res.params.items()})
        rows.append(row)
    return pd.DataFrame(rows), results
