"""Forecast alignment: each forecast is dated at the month it predicts and uses only earlier data."""
import numpy as np

from src import data as D
from src import mean_models as MM
from src import metrics as M


def test_monthly_forecasts_aligned_and_use_no_future(monkeypatch):
    close = D.make_synthetic(n=2500, seed=5)
    lp = D.monthly_log_price(close)
    r = D.monthly_returns(lp)
    fc, spec = MM.rolling_origin(lp, r, init=96, max_origins=6, log=lambda *a: None)
    h = fc[fc["model"] == "Historical mean"].reset_index(drop=True)
    # actual column equals the realised return at the stated date
    np.testing.assert_allclose(h["actual"].to_numpy(), r.loc[h["date"]].to_numpy())
    # historical-mean forecast equals the mean of strictly earlier returns
    for _, row in h.iterrows():
        prior = r[r.index < row["date"]]
        assert abs(row["pred"] - prior.mean()) < 1e-10
    # corrupting the last return must not change any earlier forecast
    r2 = r.copy()
    r2.iloc[-1] = 1e3
    lp2 = lp.copy()
    lp2.iloc[-1] = lp2.iloc[-2] + 1e3
    fc2, _ = MM.rolling_origin(lp2, r2, init=96, max_origins=6, log=lambda *a: None)
    a = fc[fc["date"] < r.index[-1]].reset_index(drop=True)
    b = fc2[fc2["date"] < r.index[-1]].reset_index(drop=True)
    np.testing.assert_allclose(a["pred"].to_numpy(), b["pred"].to_numpy(), rtol=1e-6, atol=1e-8)


def test_directional_accuracy_undefined_for_zero_drift():
    out = M.point_metrics([1.0, -1.0, 2.0], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
    assert np.isnan(out["Directional acc."])


def test_point_metrics_known_values():
    out = M.point_metrics([1.0, 2.0, 3.0], [1.0, 1.0, 5.0], [2.0, 2.0, 2.0])
    assert out["MAE"] == np.mean([0, 1, 2]) and abs(out["RMSE"] - np.sqrt(5 / 3)) < 1e-12
    assert abs(out["OOS R2 vs hist. mean"] - (1 - 5 / 2)) < 1e-12


def test_var_backtest_conditional_coverage_combines_both():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(2000)
    bt = M.var_backtest(r, np.full(2000, -2.326), 0.01)
    assert 0 <= bt["Cond. coverage p"] <= 1 and bt["Violation rate"] < 0.03
