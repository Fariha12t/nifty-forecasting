"""Mean forecasting of monthly NIFTY 50 log returns with rolling-origin (expanding window) evaluation.

Models: historical-mean and zero-drift baselines, ARIMA, SARIMA, ETS (on the log price level)
and gradient boosting on lagged features (with quantile models for prediction intervals).
Every forecast at origin i uses only information available through month i-1.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

Z95 = stats.norm.ppf(0.975)
S = 12  # seasonal period (months)
MODELS = ["Zero drift", "Historical mean", "ARIMA", "SARIMA", "ETS", "Gradient boosting"]


# ----------------------------- model selection (first training window only) -----------------------------
def _fit_sarimax(y, order, seasonal):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, order=order, seasonal_order=seasonal, trend="c").fit(disp=False)


def select_arima(y: np.ndarray, seasonal: bool) -> tuple[tuple, tuple]:
    """Pick (p,0,q) and, for SARIMA, (P,0,Q,12) by AIC. Orders are then held fixed; parameters are re-estimated at each origin."""
    best, best_aic = None, np.inf
    if seasonal:
        grid = [
            ((p, 0, q), (P, 0, Q, S))
            for p, q, P, Q in itertools.product((0, 1), (0, 1), (0, 1), (0, 1))
            if (P, Q) != (0, 0)
        ]
    else:
        grid = [((p, 0, q), (0, 0, 0, 0)) for p, q in itertools.product((0, 1, 2), (0, 1, 2))]
    for order, sorder in grid:
        try:
            aic = _fit_sarimax(y, order, sorder).aic
        except Exception:
            continue
        if np.isfinite(aic) and aic < best_aic:
            best, best_aic = (order, sorder), aic
    return best if best is not None else ((0, 0, 0), (0, 0, 0, 0))


ETS_CONFIGS = {
    "ETS(A,N,N)": dict(trend=None, damped_trend=False, seasonal=None),
    "ETS(A,Ad,N)": dict(trend="add", damped_trend=True, seasonal=None),
    "ETS(A,N,A)": dict(trend=None, damped_trend=False, seasonal="add"),
    "ETS(A,Ad,A)": dict(trend="add", damped_trend=True, seasonal="add"),
}


def _fit_ets(level: np.ndarray, cfg: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        level = pd.Series(np.asarray(level, dtype=float))  # ETS prediction needs a pandas index
        mod = ETSModel(level, error="add", seasonal_periods=S if cfg["seasonal"] else None, **cfg)
        return mod.fit(disp=False, maxiter=200)


def select_ets(level: np.ndarray) -> str:
    best, best_aic = "ETS(A,N,N)", np.inf
    for name, cfg in ETS_CONFIGS.items():
        try:
            aic = _fit_ets(level, cfg).aic
        except Exception:
            continue
        if np.isfinite(aic) and aic < best_aic:
            best, best_aic = name, aic
    return best


# ----------------------------- ML features -----------------------------
def make_features(r: pd.Series) -> pd.DataFrame:
    """Row t holds information through month t only; the target is r[t+1]."""
    f = pd.DataFrame(index=r.index)
    for k in range(1, 7):
        f[f"lag{k}"] = r.shift(k - 1)
    for w in (3, 6, 12):
        f[f"mean{w}"] = r.rolling(w).mean()
    for w in (6, 12):
        f[f"std{w}"] = r.rolling(w).std()
    target_month = (r.index.month % 12) + 1
    f["moy_sin"] = np.sin(2 * np.pi * target_month / 12)
    f["moy_cos"] = np.cos(2 * np.pi * target_month / 12)
    return f


def _gbr(**kw):
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0, **kw
    )


# ----------------------------- rolling-origin loop -----------------------------
def rolling_origin(
    lp: pd.Series,
    r: pd.Series,
    init: int = 96,
    max_origins: int | None = None,
    log=print,
) -> tuple[pd.DataFrame, dict]:
    """Expanding-window, one-step-ahead forecasts of r[i] for i = start..n-1.

    `lp` is the month-end log price x100 (length n+1), `r` its first difference (length n); r[k] = lp[k+1] - lp[k].
    Returns a long DataFrame (model, date, actual, pred, lo, hi) and the selected specifications.
    """
    n = len(r)
    start = max(init, n - max_origins) if max_origins else init
    start = max(start, 24)
    r_arr, lp_arr = r.to_numpy(), lp.to_numpy()

    order_a, _ = select_arima(r_arr[:start], seasonal=False)
    order_s, sorder_s = select_arima(r_arr[:start], seasonal=True)
    ets_name = select_ets(lp_arr[: start + 1])
    ets_cfg = ETS_CONFIGS[ets_name]
    spec = {"ARIMA": f"ARIMA{order_a} + constant", "SARIMA": f"SARIMA{order_s}x{sorder_s}", "ETS": ets_name}
    log(f"  selected on first window: {spec}")

    feats = make_features(r)
    feat_arr = feats.to_numpy()
    rows, fails = [], {"ARIMA": 0, "SARIMA": 0, "ETS": 0}

    def add(model, i, pred, lo, hi):
        rows.append((model, r.index[i], r_arr[i], pred, lo, hi))

    for i in range(start, n):
        train = r_arr[:i]
        mu_hat, sd = train.mean(), train.std(ddof=1)
        w = Z95 * sd * np.sqrt(1 + 1 / i)
        add("Zero drift", i, 0.0, -Z95 * sd, Z95 * sd)
        add("Historical mean", i, mu_hat, mu_hat - w, mu_hat + w)

        for name, order, sorder in (("ARIMA", order_a, (0, 0, 0, 0)), ("SARIMA", order_s, sorder_s)):
            try:
                fc = _fit_sarimax(train, order, sorder).get_forecast(1).summary_frame(alpha=0.05).iloc[0]
                add(name, i, fc["mean"], fc["mean_ci_lower"], fc["mean_ci_upper"])
            except Exception:
                fails[name] += 1
                add(name, i, mu_hat, mu_hat - w, mu_hat + w)

        try:
            res = _fit_ets(lp_arr[: i + 1], ets_cfg)
            fc = res.get_prediction(start=i + 1, end=i + 1).summary_frame(alpha=0.05).iloc[0]
            add("ETS", i, fc["mean"] - lp_arr[i], fc["pi_lower"] - lp_arr[i], fc["pi_upper"] - lp_arr[i])
        except Exception:
            fails["ETS"] += 1
            add("ETS", i, mu_hat, mu_hat - w, mu_hat + w)

        # ML: training rows t = 0..i-2 (target r[t+1] <= r[i-1]); predict from row i-1
        X_all = feat_arr[: i - 1]
        y_all = r_arr[1:i]
        ok = ~np.isnan(X_all).any(axis=1)
        x_new = feat_arr[i - 1 : i]
        if ok.sum() >= 24 and not np.isnan(x_new).any():
            m = _gbr(loss="squared_error").fit(X_all[ok], y_all[ok])
            lo_m = _gbr(loss="quantile", alpha=0.025).fit(X_all[ok], y_all[ok])
            hi_m = _gbr(loss="quantile", alpha=0.975).fit(X_all[ok], y_all[ok])
            p = float(m.predict(x_new)[0])
            lo, hi = float(lo_m.predict(x_new)[0]), float(hi_m.predict(x_new)[0])
            add("Gradient boosting", i, p, min(lo, p), max(hi, p))
        else:
            add("Gradient boosting", i, mu_hat, mu_hat - w, mu_hat + w)

    out = pd.DataFrame(rows, columns=["model", "date", "actual", "pred", "lo", "hi"])
    spec["fit_failures"] = fails
    n_orig = n - start
    bad = {k: v for k, v in fails.items() if v > 0.1 * n_orig}
    if bad:
        log(f"  WARNING: more than 10% of fits failed and fell back to the historical mean: {bad}")
    spec["_orders"] = {"ARIMA": (order_a, (0, 0, 0, 0)), "SARIMA": (order_s, sorder_s)}
    return out, spec


def final_arima_residuals(r: pd.Series, spec_order: tuple) -> tuple[np.ndarray, int]:
    """Full-sample ARIMA fit, for residual diagnostics only (not used for forecast evaluation)."""
    order, sorder = spec_order
    res = _fit_sarimax(r.to_numpy(), order, sorder)
    resid = np.asarray(res.resid)[max(order[0], 1):]
    return resid, order[0] + order[2]
