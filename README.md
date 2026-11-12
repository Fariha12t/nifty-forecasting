# Time-Series Forecasting and Volatility Analysis: NIFTY 50

![tests](https://github.com/Fariha12t/nifty-forecasting/actions/workflows/ci.yml/badge.svg)

Forecasting India's NIFTY 50 index two ways, with an evaluation design built to avoid the usual mistakes (look-ahead bias, in-sample model selection, accuracy numbers with no uncertainty):

1. **Mean forecasting (monthly returns):** ARIMA, SARIMA, ETS and gradient boosting against honest baselines, scored with rolling-origin cross-validation, prediction intervals and Diebold-Mariano tests.
2. **Volatility forecasting (daily returns):** GARCH, GJR-GARCH and EGARCH with Student-t errors against EWMA and rolling-window baselines, scored with QLIKE and backtested as Value-at-Risk (Kupiec and Christoffersen tests).

The point of the project is not to "beat the market". It is to show what each model can and cannot do, and to test that claim properly.

## What it covers

| Requirement | Where |
|---|---|
| Stationarity tests | ADF and KPSS on log price and returns (daily and monthly), `results/stationarity_tests.csv` |
| Models | ARIMA, SARIMA, ETS (on log price level), gradient boosting (lag features), GARCH / GJR-GARCH / EGARCH |
| Baselines | Zero drift, expanding historical mean, EWMA (RiskMetrics 0.94), rolling 250-day variance |
| Residual diagnostics | Residual plot, histogram, Q-Q, ACF, Ljung-Box (including squared standardised residuals for GARCH), ARCH-LM |
| Rolling-origin cross-validation | Expanding window, one-step ahead, parameters re-estimated at every origin (monthly) or every 21 days (daily) |
| Prediction intervals | 95% intervals for every model, scored on coverage, width and Winkler score |
| Model comparison | Diebold-Mariano with Newey-West variance and small-sample correction |
| Risk | 99% and 95% one-day VaR with Kupiec, independence and conditional-coverage backtests |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.run_all --data nifty --out results
```

A full run takes roughly 10 to 20 minutes. It downloads NIFTY 50 (`^NSEI`) from Yahoo Finance. If the download fails, export a `Date,Close` CSV and run:

```bash
python -m src.run_all --data nifty --csv nifty.csv --out results
```

Other options:

```bash
python -m src.run_all --quick --data synthetic   # 20-second smoke test on fake data (never report these numbers)
python -m src.run_all --max-origins 60           # only the most recent 60 monthly forecasts
python -m pytest -q                              # run the tests
```

## Outputs

Running the pipeline writes everything to `results/`, including **`REPORT.md`**, an auto-generated report with every table and a plain-language headline for each section. Figures are in `results/figures/`:

`overview.png`, `acf_pacf_daily_returns.png`, `monthly_forecast_intervals_best.png`, `monthly_cum_sse_vs_benchmark.png`, `monthly_rmse.png`, `residuals_arima.png`, `residuals_sarima.png`, `residuals_garch.png`, `volatility_forecasts.png`, `var_violations_best.png`

## Method notes

- **No look-ahead.** Every forecast at origin *i* uses data through *i-1* only. `tests/test_no_lookahead.py` checks this by corrupting all future data and confirming the forecasts do not change.
- **Model selection** (ARIMA/SARIMA order, ETS form) uses AIC on the first training window only, so selection never sees the evaluation period. Parameters are re-estimated at every monthly origin.
- **ARIMA** is fitted to monthly log returns with a constant, which is equivalent to ARIMA(p,1,q) with drift on the log price. **SARIMA** adds a 12-month seasonal component. **ETS** models the log price level and the forecast is converted back to a return.
- **Gradient boosting** uses 6 return lags, rolling means and volatilities, and month-of-year terms. Intervals come from quantile (2.5% and 97.5%) boosted models.
- **Volatility** models use Student-t errors, so the VaR quantiles reflect fat tails. The variance proxy is the squared return, which is noisy; QLIKE is used because it ranks forecasts correctly despite that noise.
- If a model fit fails, the forecast falls back to the historical mean and the failure is counted and shown in the report, never hidden.

## How to read the results

- A negative **OOS R²** means the model did worse than simply predicting the historical average return. For index returns this is common and is itself an informative result.
- **DM p < 0.05** against the benchmark means the difference in squared error is statistically distinguishable from zero.
- **Coverage near 95%** means the prediction intervals are honest. Much lower means the model is overconfident.
- For VaR, a **Kupiec p above 0.05** means the violation rate is consistent with the target; the **independence p** tests whether violations cluster in crises.

## Results

See [`results/REPORT.md`](results/REPORT.md) after running the pipeline. Add a short summary of your own findings here, for example which model won on monthly RMSE (and whether any beat the historical mean), which volatility model had the lowest QLIKE, and which VaR models passed the backtests.

## Limitations

- One index, one sample period. Conclusions may not transfer to other markets or regimes.
- Orders are chosen once on the first window, so structural changes later in the sample are not re-selected.
- Daily GARCH parameters are re-estimated every 21 trading days for speed, not every day.
- Nothing here is investment advice.

## Project layout

```
src/data.py          NIFTY download / CSV / synthetic test data
src/stats_tests.py   ADF, KPSS, Ljung-Box, ARCH-LM, descriptives
src/mean_models.py   ARIMA, SARIMA, ETS, gradient boosting, rolling-origin loop
src/volatility.py    GARCH family, EWMA, rolling-window, quantiles
src/metrics.py       accuracy, intervals, Diebold-Mariano, QLIKE, VaR backtests
src/plots.py         figures
src/run_all.py       pipeline and REPORT.md generator
tests/               unit tests, look-ahead guards, end-to-end smoke test
```


