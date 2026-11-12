"""Stationarity tests and descriptive diagnostics."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_table(series: dict[str, pd.Series], alpha: float = 0.05) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) side by side, with a plain-English verdict."""
    rows = []
    for name, s in series.items():
        x = s.dropna().to_numpy()
        adf_stat, adf_p = adfuller(x, autolag="AIC")[:2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # KPSS warns when p is outside its lookup table
            kpss_stat, kpss_p = kpss(x, regression="c", nlags="auto")[:2]
        if adf_p < alpha and kpss_p > alpha:
            verdict = "stationary"
        elif adf_p >= alpha and kpss_p <= alpha:
            verdict = "non-stationary (unit root)"
        else:
            verdict = "inconclusive (tests disagree)"
        rows.append(
            {
                "series": name,
                "n": len(x),
                "ADF stat": adf_stat,
                "ADF p": adf_p,
                "KPSS stat": kpss_stat,
                "KPSS p": kpss_p,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def descriptives(r: pd.Series, lags: int = 10) -> pd.DataFrame:
    """Moments, normality and dependence tests for a return series."""
    x = r.dropna().to_numpy()
    jb_stat, jb_p = stats.jarque_bera(x)
    lb_r = acorr_ljungbox(x, lags=[lags], return_df=True).iloc[0]
    lb_r2 = acorr_ljungbox(x**2, lags=[lags], return_df=True).iloc[0]
    arch_lm = het_arch(x - x.mean(), nlags=lags)
    out = {
        "n": len(x),
        "mean": x.mean(),
        "std": x.std(ddof=1),
        "skew": stats.skew(x),
        "excess kurtosis": stats.kurtosis(x),
        "Jarque-Bera p": jb_p,
        f"Ljung-Box({lags}) p, returns": lb_r["lb_pvalue"],
        f"Ljung-Box({lags}) p, squared returns": lb_r2["lb_pvalue"],
        f"ARCH-LM({lags}) p": arch_lm[1],
    }
    return pd.DataFrame({"value": out})


def ljung_box_p(resid, lags: int = 12, model_df: int = 0) -> float:
    x = np.asarray(resid, dtype=float)
    x = x[~np.isnan(x)]
    lags = min(lags, max(len(x) // 5, 1))
    res = acorr_ljungbox(x, lags=[lags], model_df=model_df, return_df=True)
    return float(res["lb_pvalue"].iloc[0])
