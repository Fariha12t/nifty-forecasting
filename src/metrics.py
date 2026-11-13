"""Forecast accuracy, interval quality, Diebold-Mariano test, volatility losses and VaR backtests."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import xlogy


# ----------------------------- point and interval forecasts -----------------------------
def point_metrics(actual, pred, bench_pred) -> dict:
    actual, pred, bench_pred = map(np.asarray, (actual, pred, bench_pred))
    e, eb = actual - pred, actual - bench_pred
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "OOS R2 vs hist. mean": float(1 - np.sum(e**2) / np.sum(eb**2)),
        # A forecast of exactly 0 has no direction, so directional accuracy is undefined (NaN), not 0%.
        "Directional acc.": float(np.mean(np.sign(pred) == np.sign(actual))) if np.any(pred != 0) else float("nan"),
    }


def interval_metrics(actual, lo, hi, level: float = 0.95) -> dict:
    actual, lo, hi = map(np.asarray, (actual, lo, hi))
    alpha = 1 - level
    width = hi - lo
    below, above = actual < lo, actual > hi
    winkler = width + (2 / alpha) * (lo - actual) * below + (2 / alpha) * (actual - hi) * above
    return {
        f"{int(level*100)}% coverage": float(np.mean(~(below | above))),
        "Mean width": float(width.mean()),
        "Winkler score": float(winkler.mean()),
    }


def nw_lag(n: int) -> int:
    """Rule-of-thumb Newey-West truncation lag, floor(n^(1/3))."""
    return int(np.floor(n ** (1 / 3)))


def dm_test(loss_a, loss_b, h: int = 1, max_lag: int | None = None):
    """Diebold-Mariano test with Newey-West variance and the Harvey-Leybourne-Newbold correction.

    Negative statistic: model A has the lower average loss. Returns (statistic, two-sided p-value).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if max_lag is None:
        max_lag = h - 1
    dbar = d.mean()
    dc = d - dbar
    lrv = dc @ dc / n
    for k in range(1, max_lag + 1):
        w = 1 - k / (max_lag + 1)
        lrv += 2 * w * (dc[k:] @ dc[:-k]) / n
    lrv = max(lrv, 1e-16)
    stat = dbar / np.sqrt(lrv / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


# ----------------------------- volatility -----------------------------
def vol_point_metrics(sigma2, proxy) -> dict:
    """Variance-forecast errors against a noisy proxy (squared return). Lower is better for all three."""
    sigma2, proxy = np.asarray(sigma2, dtype=float), np.asarray(proxy, dtype=float)
    return {
        "MSE (variance)": float(np.mean((sigma2 - proxy) ** 2)),
        "MAE (sigma vs |r|)": float(np.mean(np.abs(np.sqrt(sigma2) - np.sqrt(proxy)))),
        "QLIKE": float(np.mean(qlike(sigma2, proxy))),
    }


def qlike(sigma2, proxy):
    """QLIKE loss (up to a constant): robust to noise in the variance proxy (Patton, 2011)."""
    sigma2 = np.maximum(np.asarray(sigma2, dtype=float), 1e-10)
    return np.log(sigma2) + np.asarray(proxy) / sigma2


def kupiec_pof(viol, p: float):
    """Kupiec proportion-of-failures test. Returns (LR statistic, p-value)."""
    viol = np.asarray(viol).astype(int)
    n, x = len(viol), int(viol.sum())
    pihat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - pihat) + xlogy(x, pihat)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_ind(viol):
    """Christoffersen independence test: are violations clustered? Returns (LR, p-value)."""
    v = np.asarray(viol).astype(int)
    a, b = v[:-1], v[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    tot = n00 + n01 + n10 + n11
    pi = (n01 + n11) / tot
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = (
        xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    )
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def var_backtest(r, var_level, p: float) -> dict:
    """Violation rate plus Kupiec, Christoffersen and conditional-coverage p-values."""
    r, var_level = np.asarray(r), np.asarray(var_level)
    viol = r < var_level
    lr_pof, p_pof = kupiec_pof(viol, p)
    lr_ind, p_ind = christoffersen_ind(viol)
    p_cc = float(1 - stats.chi2.cdf(lr_pof + lr_ind, 2))
    return {
        "Violation rate": float(viol.mean()),
        "Expected": p,
        "Kupiec p": p_pof,
        "Independence p": p_ind,
        "Cond. coverage p": p_cc,
    }
