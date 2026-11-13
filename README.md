# Forecasting and Volatility Analysis of the NIFTY 50

![tests](https://github.com/Fariha12t/nifty-forecasting/actions/workflows/ci.yml/badge.svg)

A reproducible time-series study of India's NIFTY 50 index, built to answer two questions honestly rather than to "beat the market".

## 1. Project overview

The project forecasts the NIFTY 50 in two ways and tests each claim statistically:

1. **Monthly return forecasting** — do ARIMA, SARIMA, ETS or gradient boosting predict next month's return better than a no-skill baseline?
2. **Daily volatility forecasting and risk** — do GARCH-family models forecast volatility and Value-at-Risk better than simple EWMA and rolling-window estimators?

What it demonstrates is the evaluation discipline: strictly out-of-sample, expanding-window forecasts; model orders chosen only on the first training window; every fallback and failure counted; and formal tests (Diebold–Mariano, Kupiec, Christoffersen) instead of eyeballed accuracy tables.

## 2. Research questions

- **RQ1.** Does any model beat the zero-drift and expanding-historical-mean baselines for monthly returns, in squared error and direction, and is the difference statistically significant after multiple-testing adjustment?
- **RQ2.** Are NIFTY 50 returns stationary and do they show volatility clustering (ADF, KPSS, Ljung–Box, ARCH-LM)?
- **RQ3.** Do asymmetric, fat-tailed GARCH models (GJR, EGARCH, Student-t) forecast volatility (QLIKE) better than EWMA (λ = 0.94) and a rolling 250-day variance?
- **RQ4.** Which models produce 95% and 99% VaR forecasts that pass Kupiec, independence and conditional-coverage backtests?

## 3. Dataset and source

Daily NIFTY 50 closing prices, Yahoo Finance ticker `^NSEI` (unadjusted close), from 2007-09-17 (the first date Yahoo provides). The run records the exact series used (`results/nifty_close_used.csv`), a data-quality report (`results/data_validation.json`: missing values, duplicate dates, non-positive prices, ordering, gaps) and the download date. Non-trading days are not filled; returns are between consecutive trading days. If Yahoo is unavailable, supply a `Date,Close` CSV with `--csv`. The pipeline never substitutes synthetic data for results.

## 4. Statistical methodology

- Log returns r_t = 100·(ln P_t − ln P_{t−1}), daily; monthly returns from month-end closes (a partial trailing month is dropped).
- Stationarity: ADF and KPSS on log prices and returns, reported together with a combined verdict. Distribution: skewness, excess kurtosis, Jarque–Bera, histogram/density, Q–Q plot. Dependence: ACF/PACF, Ljung–Box on returns, squared returns and absolute returns, ARCH-LM at several lags.
- Residual diagnostics for ARIMA, SARIMA and the best GARCH model (in-sample, full-sample fits; used for diagnosis only, never for forecast evaluation).

## 5. Forecasting framework (monthly)

Expanding window, one step ahead, first training window of 96 months, parameters re-estimated at every origin.

| Model | Specification |
|---|---|
| Zero drift | forecast 0 |
| Historical mean | mean of all earlier returns |
| ARIMA | (p,0,q), p,q ≤ 2, with constant; order by AIC on the first window only |
| SARIMA | adds (P,0,Q,12); order by AIC on the first window only |
| ETS | additive-error exponential smoothing on log price, form by AIC on first window; forecast converted to a return |
| Gradient boosting | 6 lags, rolling means/std, month-of-year terms; 95% intervals from 2.5%/97.5% quantile models |

Metrics: MAE, RMSE, out-of-sample R² vs the historical mean, directional accuracy (undefined for zero drift), 95% interval coverage, mean width, Winkler score. Pairwise Diebold–Mariano tests on squared error with Newey–West variance (lag ⌊n^(1/3)⌋) and the Harvey–Leybourne–Newbold small-sample correction, with Holm adjustment across pairs. Statistical significance is reported separately from the size of the improvement.

## 6. Volatility modelling framework (daily)

GARCH(1,1), GJR-GARCH(1,1) and EGARCH(1,1) with Student-t innovations and a constant mean, re-estimated every 21 trading days on data strictly before each block, one-step-ahead variance forecasts with parameters held fixed inside a block. Baselines: EWMA (λ = 0.94) and rolling 250-day variance. If a re-estimation fails, the last successful parameters are reused and the event is logged in `results/fallback_events.csv` (and counted in the report); optimiser non-convergence is logged too.

Squared returns are the variance proxy. They are unbiased but extremely noisy, so absolute loss levels are not meaningful; QLIKE is used because it ranks forecasts consistently under that noise (Patton, 2011). Intraday realised variance would be a better proxy but is not available here.

## 7. Evaluation metrics

QLIKE, variance MSE and σ-vs-|r| MAE; pairwise DM tests on QLIKE; 95% and 99% one-day VaR with the Kupiec unconditional-coverage test, the Christoffersen independence test and the combined conditional-coverage test. Baseline VaR uses a normal quantile while GARCH-family VaR uses the fitted t, so VaR differences mix variance and tail assumptions.

## 8. Actual numerical findings

<!-- RESULTS:START -->
_Not yet generated. The tables here are written automatically by `python -m src.run_all --data nifty` from real NIFTY 50 data. No numbers are shown until that run has been done._
<!-- RESULTS:END -->

## 9. Visualizations

Saved to `results/figures/` by the run: price/returns/volatility overview, monthly returns, return distribution + Q–Q, daily and monthly ACF/PACF, volatility clustering, monthly actual-vs-forecast (all models), 95% prediction intervals, cumulative SSE vs benchmark, RMSE comparison, ARIMA/SARIMA/GARCH residual diagnostics, GARCH conditional volatility, volatility forecasts vs |return|, and 99%/95% VaR exceedance plots.

## 10. Limitations

- One index, one sample period; results may not transfer to other markets or regimes.
- ARIMA/SARIMA/ETS orders are chosen once on the first window and then frozen.
- GARCH parameters are re-estimated every 21 days, not daily.
- Squared returns are a noisy volatility proxy.
- Many pairwise tests: rely on Holm-adjusted p-values; with ~100–200 monthly forecasts, power is low, so "not significant" is not "equally good".
- Residual diagnostics are in-sample. Nothing here is investment advice.

## 11. Installation and execution

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                                   # unit tests (synthetic data, no network)
python -m src.run_all --data nifty --out results      # downloads ^NSEI, runs everything (~10-20 min)
# or, offline: python -m src.run_all --data nifty --csv data/nifty.csv --out results
```

`./reproduce.sh` does all of the above in a fresh environment. Never report output from `--data synthetic`; that mode exists only to test the code.

## 12. Repository structure

```
src/data.py          download/CSV loading, validation, return construction
src/stats_tests.py   ADF, KPSS, Ljung-Box, ARCH-LM, clustering tests
src/mean_models.py   baselines, ARIMA, SARIMA, ETS, gradient boosting, rolling-origin loop
src/volatility.py    GARCH family, EWMA, rolling window, fallback logging
src/metrics.py       accuracy, intervals, Diebold-Mariano, QLIKE, VaR backtests
src/plots.py         figures
src/run_all.py       pipeline, statistical_tests.csv, REPORT.md, README results block
tests/               returns, alignment, look-ahead, metrics, data validation, end-to-end
results/             generated outputs (CSVs, REPORT.md, figures/)
```
