import numpy as np
import pandas as pd
import pytest

from src import data as D


def _close():
    idx = pd.to_datetime(["2020-01-29", "2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02", "2020-03-31"])
    return pd.Series([100.0, 101.0, 99.0, 100.0, 110.0, 108.0, 120.0], index=idx, name="close")


def test_daily_log_return_values():
    c = _close()
    r = D.daily_returns(c)
    assert len(r) == len(c) - 1
    assert r.iloc[0] == pytest.approx(100 * np.log(101 / 100))
    assert r.index[0] == c.index[1]  # return is dated at the later price


def test_monthly_returns_use_month_end_and_align():
    lp = D.monthly_log_price(_close())
    r = D.monthly_returns(lp)
    # month ends: Jan 99, Feb 110, Mar 120 -> two monthly returns, dated at the later month
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(100 * np.log(110 / 99))
    assert r.iloc[1] == pytest.approx(100 * np.log(120 / 110))
    assert r.index[0].month == 2


def test_monthly_returns_sum_to_total_log_return():
    c = D.make_synthetic(n=1500, seed=1)
    r = D.monthly_returns(D.monthly_log_price(c))
    lp = D.monthly_log_price(c)
    assert r.sum() == pytest.approx(lp.iloc[-1] - lp.iloc[0])


def test_validate_prices_flags_problems():
    c = _close()
    bad = pd.concat([c, c.iloc[[0]]])
    bad.iloc[1] = np.nan
    rep = D.validate_prices(bad)
    assert rep["n_duplicate_dates"] == 1 and rep["n_missing_values"] == 1 and not rep["is_monotonic_increasing_dates"]


def test_nonpositive_prices_raise(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("Date,Close\n2020-01-01,100\n2020-01-02,0\n2020-01-03,101\n")
    with pytest.raises(ValueError):
        D.load_nifty(csv=str(f))


def test_csv_cleaning_sorts_and_dedups(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("Date,Close\n2020-01-03,101\n2020-01-01,100\n2020-01-01,100\n2020-01-02,\n")
    c, rep = D.load_nifty(csv=str(f), return_report=True)
    assert list(c.index) == sorted(c.index) and c.index.is_unique and len(c) == 2
    assert rep["n_duplicate_dates"] == 1
