"""Data loading: NIFTY 50 from Yahoo Finance (^NSEI), a CSV, or a synthetic stand-in for offline tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_prices(close: pd.Series) -> dict:
    """Data-quality report on a raw close series (before cleaning). Never modifies the data."""
    idx = pd.DatetimeIndex(close.index)
    gaps = idx.to_series().diff().dt.days.dropna()
    return {
        "n_raw": int(len(close)),
        "n_missing_values": int(close.isna().sum()),
        "n_duplicate_dates": int(idx.duplicated().sum()),
        "n_nonpositive_prices": int((close <= 0).sum()),
        "is_monotonic_increasing_dates": bool(idx.is_monotonic_increasing),
        "n_weekend_dates": int((idx.dayofweek >= 5).sum()),
        "max_gap_calendar_days": int(gaps.max()) if len(gaps) else 0,
        "n_gaps_over_5_days": int((gaps > 5).sum()),
    }


def load_nifty(
    start: str = "2007-09-17", end: str | None = None, csv: str | None = None, return_report: bool = False
):
    """Daily NIFTY 50 close prices as a float Series with a tz-naive DatetimeIndex.

    Source is Yahoo Finance (^NSEI, unadjusted close; an index has no dividends/splits to adjust) or a user CSV.
    Cleaning: drop NaN closes, drop duplicated dates (keep first), sort chronologically. Non-trading days are
    NOT forward-filled: returns are computed between consecutive trading days. Non-positive prices raise.
    """
    if csv:
        df = pd.read_csv(csv, parse_dates=[0], index_col=0)
        close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        source = f"csv:{csv}"
    else:
        import yfinance as yf

        df = yf.download("^NSEI", start=start, end=end, auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError("No data returned from Yahoo Finance. Check your connection or pass --csv.")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        source = "yahoo_finance:^NSEI"
    close = close.astype(float)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    report = validate_prices(close)
    report["source"] = source
    close = close.dropna()
    close = close[~close.index.duplicated()].sort_index()
    if (close <= 0).any():
        raise ValueError(f"{int((close <= 0).sum())} non-positive prices found; refusing to take logs. Inspect the data.")
    close.name = "close"
    report.update(
        n_clean=int(len(close)),
        first_date=str(close.index[0].date()),
        last_date=str(close.index[-1].date()),
        download_date_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
    )
    return (close, report) if return_report else close


def make_synthetic(n: int = 4000, seed: int = 0) -> pd.Series:
    """GJR-GARCH(1,1) with Student-t shocks. For offline smoke tests only; never report its numbers."""
    rng = np.random.default_rng(seed)
    omega, alpha, gamma, beta, nu, mu = 0.02, 0.04, 0.08, 0.90, 6.0, 0.05
    scale = np.sqrt((nu - 2) / nu)
    r = np.empty(n)
    s2 = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        z = rng.standard_t(nu) * scale
        r[t] = mu + np.sqrt(s2) * z
        e = r[t] - mu
        s2 = omega + (alpha + gamma * (e < 0)) * e**2 + beta * s2
    idx = pd.bdate_range("2007-09-17", periods=n)
    close = pd.Series(1000 * np.exp(np.cumsum(r) / 100), index=idx, name="close")
    return close


def daily_returns(close: pd.Series) -> pd.Series:
    """Daily log returns in percent."""
    return (np.log(close).diff().dropna() * 100).rename("ret")


def monthly_log_price(close: pd.Series) -> pd.Series:
    """Month-end log price x 100. A trailing partial month is dropped."""
    try:
        m = close.resample("ME").last()
    except ValueError:  # older pandas
        m = close.resample("M").last()
    last = close.index[-1]
    next_bday = last + pd.offsets.BDay(1)
    if next_bday.month == last.month:  # last observed month is incomplete
        m = m.iloc[:-1]
    return (np.log(m.dropna()) * 100).rename("lp100")


def monthly_returns(lp100: pd.Series) -> pd.Series:
    """Monthly log returns in percent; r[k] = lp[k+1] - lp[k], indexed by the later date."""
    return lp100.diff().dropna().rename("ret_m")
