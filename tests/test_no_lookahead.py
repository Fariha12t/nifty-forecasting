"""Guards against look-ahead bias, the most common mistake in forecast evaluation."""
import numpy as np

from src import data as D
from src import mean_models as MM
from src import volatility as V


def _returns(n=1500):
    return D.daily_returns(D.make_synthetic(n=n, seed=3))


def test_garch_forecast_ignores_future_data():
    y = _returns()
    t, k = 1000, 10
    spec = V.SPECS["GJR-GARCH(1,1)-t"]
    var_full, mu_f, nu_f = V.garch_forecast_block(y, t, k, spec)
    # Truncate everything the forecasts should not depend on, and corrupt it for good measure.
    y_short = y.iloc[: t + k - 1]
    var_short, mu_s, nu_s = V.garch_forecast_block(y_short, t, k, spec)
    np.testing.assert_allclose(var_full, var_short, rtol=1e-8)
    y_bad = y.copy()
    y_bad.iloc[t + k - 1 :] = 50.0
    var_bad, _, _ = V.garch_forecast_block(y_bad, t, k, spec)
    np.testing.assert_allclose(var_full, var_bad, rtol=1e-8)


def test_ewma_and_rolling_use_only_past():
    y = _returns()
    e1 = V.ewma_variance(y)
    y2 = y.copy()
    y2.iloc[1200:] = 99.0
    e2 = V.ewma_variance(y2)
    np.testing.assert_allclose(e1.iloc[:1201], e2.iloc[:1201])  # day 1200's forecast uses returns through 1199
    r1 = V.rolling_window_variance(y)
    r2 = V.rolling_window_variance(y2)
    np.testing.assert_allclose(r1.iloc[:1201].dropna(), r2.iloc[:1201].dropna())


def test_ml_features_only_use_past():
    r = D.monthly_returns(D.monthly_log_price(D.make_synthetic(n=3000, seed=4)))
    f1 = MM.make_features(r)
    r2 = r.copy()
    r2.iloc[100:] = 999.0
    f2 = MM.make_features(r2)
    np.testing.assert_allclose(f1.iloc[:100].to_numpy(), f2.iloc[:100].to_numpy(), equal_nan=True)
