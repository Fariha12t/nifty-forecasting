# Results report (auto-generated)



## 1. Dataset and methodology

- Source: **NIFTY 50 (^NSEI, Yahoo Finance)**; downloaded/loaded on 2026-11-13.
- Daily sample: 2007-09-17 to 2026-11-13, 4700 closing prices, 4699 daily log returns (percent).
- Monthly sample: 229 month-end returns (partial trailing month dropped).
- Data validation: {"n_raw": 4700, "n_missing_values": 0, "n_duplicate_dates": 0, "n_nonpositive_prices": 0, "is_monotonic_increasing_dates": true, "n_weekend_dates": 1, "max_gap_calendar_days": 6, "n_gaps_over_5_days": 1, "n_clean": 4700}.
- Cleaning: NaN closes and duplicate dates dropped, sorted chronologically; non-trading days are not filled; non-positive prices raise an error.
- Log returns: r_t = 100·(ln P_t − ln P_{t−1}); monthly returns use month-end closes.

## 2. Forecasting design

- Monthly: expanding window, one-step ahead, first training window 96 months, **133 out-of-sample forecasts**, models re-estimated at every origin.
- Selected on the first window only: ARIMA(0, 0, 0) + constant; SARIMA(0, 0, 0)x(1, 0, 1, 12); ETS(A,N,N).
- Daily: GARCH-family models (Student-t, constant mean) re-estimated every 21 trading days on data strictly before each block; **3199 out-of-sample daily forecasts** on common dates; EWMA λ=0.94 and rolling-250d baselines.
- Baseline VaR quantiles use a normal distribution (EWMA and rolling window have no fitted tail parameter); GARCH-family VaR uses the fitted standardised Student-t. Differences in VaR performance therefore mix variance forecasts and tail assumptions.

## 3. Diagnostics

- Log price (daily): ADF p=0.9250, KPSS p=0.0100 → **non-stationary (unit root)**.
- Daily log returns: ADF p=0.0000, KPSS p=0.1000 → **stationary**.
- Monthly log price: ADF p=0.8997, KPSS p=0.0100 → **non-stationary (unit root)**.
- Monthly log returns: ADF p=0.0000, KPSS p=0.1000 → **stationary**.
- Volatility clustering: every Ljung-Box (squared returns) and ARCH-LM test rejects the null at 5%, so the data show statistically significant ARCH effects.

| series              |    n |   ADF stat |   ADF p |   KPSS stat |   KPSS p | verdict                    |
|:--------------------|-----:|-----------:|--------:|------------:|---------:|:---------------------------|
| Log price (daily)   | 4700 |    -0.3033 |  0.9250 |     10.8228 |   0.0100 | non-stationary (unit root) |
| Daily log returns   | 4699 |   -14.6041 |  0.0000 |      0.0387 |   0.1000 | stationary                 |
| Monthly log price   |  230 |    -0.4595 |  0.8997 |      2.2962 |   0.0100 | non-stationary (unit root) |
| Monthly log returns |  229 |   -15.3777 |  0.0000 |      0.0573 |   0.1000 | stationary                 |

Daily return descriptives:

| statistic                        |         value |
|:---------------------------------|--------------:|
| n                                | 4699          |
| mean                             |    0.03664    |
| std                              |    1.292      |
| skew                             |   -0.2789     |
| excess kurtosis                  |   15.15       |
| Jarque-Bera p                    |    0          |
| Ljung-Box(10) p, returns         |    2.503e-05  |
| Ljung-Box(10) p, squared returns |    0          |
| ARCH-LM(10) p                    |    1.215e-135 |

Residual diagnostics (full-sample fits, in-sample; not forecast evaluation):

| series                               | test          |   statistic |   p_value |
|:-------------------------------------|:--------------|------------:|----------:|
| ARIMA residuals                      | Ljung-Box(6)  |       5.646 | 0.4639    |
| ARIMA residuals                      | Ljung-Box(12) |      11.46  | 0.4898    |
| ARIMA residuals                      | Ljung-Box(24) |      20.02  | 0.6956    |
| ARIMA residuals                      | ARCH-LM(6)    |      12.06  | 0.06071   |
| SARIMA residuals                     | Ljung-Box(6)  |       7.267 | 0.2969    |
| SARIMA residuals                     | Ljung-Box(12) |      13.96  | 0.303     |
| SARIMA residuals                     | Ljung-Box(24) |      25.12  | 0.3991    |
| SARIMA residuals                     | ARCH-LM(6)    |      15.57  | 0.01626   |
| EGARCH(1,1)-t std. residuals         | Ljung-Box(5)  |      22.63  | 0.0003968 |
| EGARCH(1,1)-t std. residuals         | Ljung-Box(10) |      28.93  | 0.001278  |
| EGARCH(1,1)-t std. residuals         | Ljung-Box(20) |      37.73  | 0.009561  |
| EGARCH(1,1)-t squared std. residuals | Ljung-Box(5)  |      11.89  | 0.03639   |
| EGARCH(1,1)-t squared std. residuals | Ljung-Box(10) |      15.3   | 0.1216    |
| EGARCH(1,1)-t squared std. residuals | Ljung-Box(20) |      22.23  | 0.328     |
| EGARCH(1,1)-t std. residuals         | ARCH-LM(10)   |      15.82  | 0.1049    |

## 4. Monthly return forecasts

| model             |   n_forecasts |   RMSE |    MAE |   OOS R2 vs hist. mean |   Directional acc. |   95% coverage |   Mean width |   Winkler score |   DM stat vs hist. mean |   DM p vs hist. mean |
|:------------------|--------------:|-------:|-------:|-----------------------:|-------------------:|---------------:|-------------:|----------------:|------------------------:|---------------------:|
| Zero drift        |           133 | 4.7943 | 3.4798 |                -0.0200 |           nan      |         0.9774 |      25.5695 |         30.0657 |                  1.0319 |               0.3040 |
| Historical mean   |           133 | 4.7472 | 3.4279 |                 0.0000 |             0.5714 |         0.9774 |      25.6543 |         30.3905 |                nan      |             nan      |
| ARIMA             |           133 | 4.7472 | 3.4279 |                 0.0000 |             0.5714 |         0.9774 |      25.4844 |         30.2878 |                 -0.0000 |               1.0000 |
| SARIMA            |           133 | 4.8097 | 3.5053 |                -0.0265 |             0.5263 |         0.9774 |      25.2148 |         30.0063 |                  1.1071 |               0.2703 |
| ETS               |           133 | 4.7952 | 3.4801 |                -0.0203 |             0.4737 |         0.9774 |      25.5247 |         30.0420 |                  1.0423 |               0.2992 |
| Gradient boosting |           133 | 5.5788 | 3.9542 |                -0.3810 |             0.4812 |         0.9398 |      15.4556 |         26.5529 |                  1.8727 |               0.0633 |

**Headline:** lowest RMSE is Historical mean (4.7472). No model's squared error is significantly different from the historical-mean benchmark (all DM p ≥ 0.05). After Holm adjustment across all 15 pairwise comparisons, 0 pair(s) remain significant at 5%. OOS R² is relative to the expanding historical-mean forecast (negative = worse than that benchmark). Directional accuracy is undefined for zero drift (it never predicts a sign). A statistically significant DM result is not by itself economically meaningful: compare the size of the RMSE/MAE gap to the monthly return scale.

Fit failures (fell back to the historical mean forecast): {'ARIMA': 0, 'SARIMA': 0, 'ETS': 0}. Statsmodels warnings counted: {'sarimax': 0, 'ets': 0}.

## 5. Daily volatility forecasts

| model            |   loglik |       AIC |       BIC |      mu |    omega |   alpha[1] |   beta[1] |    nu |   gamma[1] |
|:-----------------|---------:|----------:|----------:|--------:|---------:|-----------:|----------:|------:|-----------:|
| GARCH(1,1)-t     |    -6682 | 1.337e+04 | 1.341e+04 | 0.06607 | 0.01583  |    0.0796  |    0.9089 | 7.206 |  nan       |
| GJR-GARCH(1,1)-t |    -6630 | 1.327e+04 | 1.331e+04 | 0.04672 | 0.01947  |    0.01324 |    0.9057 | 7.636 |    0.1287  |
| EGARCH(1,1)-t    |    -6625 | 1.326e+04 | 1.33e+04  | 0.04597 | 0.003409 |    0.1423  |    0.9849 | 7.613 |   -0.09046 |

| model            |   n_forecasts |   MSE (variance) |   MAE (sigma vs |r|) |   QLIKE |   Mean forecast vol (%/day) |   95% return-interval coverage |
|:-----------------|--------------:|-----------------:|---------------------:|--------:|----------------------------:|-------------------------------:|
| GARCH(1,1)-t     |          3199 |          18.7165 |               0.5424 |  0.7279 |                      0.9557 |                         0.9609 |
| GJR-GARCH(1,1)-t |          3199 |          18.5932 |               0.5442 |  0.7102 |                      0.9749 |                         0.9619 |
| EGARCH(1,1)-t    |          3199 |          17.7499 |               0.5315 |  0.7093 |                      0.9534 |                         0.9594 |
| EWMA (0.94)      |          3199 |          19.1185 |               0.5260 |  0.7498 |                      0.9032 |                         0.9419 |
| Rolling 250d     |          3199 |          21.4316 |               0.5847 |  0.9241 |                      0.9691 |                         0.9484 |

**Headline:** lowest QLIKE is EGARCH(1,1)-t (0.7093). Lower is better. Pairwise DM on QLIKE: 7 of 10 pairs significant at 5% unadjusted, 6 after Holm adjustment.

GARCH fallback/failure events: none. Full log: `fallback_events.csv`.

## 6. Value-at-Risk backtests

| model            | VaR level   |    n |   Exceedances |   Violation rate |   Expected |   Kupiec p |   Independence p |   Cond. coverage p |
|:-----------------|:------------|-----:|--------------:|-----------------:|-----------:|-----------:|-----------------:|-------------------:|
| GARCH(1,1)-t     | 99%         | 3199 |            39 |           0.0122 |     0.0100 |     0.2285 |           0.5014 |             0.3863 |
| GARCH(1,1)-t     | 95%         | 3199 |           181 |           0.0566 |     0.0500 |     0.0941 |           0.3788 |             0.1672 |
| GJR-GARCH(1,1)-t | 99%         | 3199 |            34 |           0.0106 |     0.0100 |     0.7237 |           0.3926 |             0.6519 |
| GJR-GARCH(1,1)-t | 95%         | 3199 |           156 |           0.0488 |     0.0500 |     0.7477 |           0.3827 |             0.6487 |
| EGARCH(1,1)-t    | 99%         | 3199 |            36 |           0.0113 |     0.0100 |     0.4848 |           0.3653 |             0.5200 |
| EGARCH(1,1)-t    | 95%         | 3199 |           155 |           0.0485 |     0.0500 |     0.6865 |           0.8531 |             0.9061 |
| EWMA (0.94)      | 99%         | 3199 |            60 |           0.0188 |     0.0100 |     0.0000 |           0.9021 |             0.0001 |
| EWMA (0.94)      | 95%         | 3199 |           184 |           0.0575 |     0.0500 |     0.0565 |           0.6510 |             0.1464 |
| Rolling 250d     | 99%         | 3199 |            57 |           0.0178 |     0.0100 |     0.0001 |           0.0211 |             0.0000 |
| Rolling 250d     | 95%         | 3199 |           151 |           0.0472 |     0.0500 |     0.4638 |           0.0008 |             0.0028 |

Models not rejected by the conditional-coverage test at 5%: 99% VaR: GARCH(1,1)-t, GJR-GARCH(1,1)-t, EGARCH(1,1)-t; 95% VaR: GARCH(1,1)-t, GJR-GARCH(1,1)-t, EGARCH(1,1)-t, EWMA (0.94). Kupiec tests unconditional coverage, Christoffersen tests independence of exceedances, and conditional coverage combines them (χ²₂).

## 7. Limitations

- Single index, single sample period; conclusions may not transfer to other markets or regimes.
- ARIMA/SARIMA/ETS specifications are selected once by AIC on the first training window and then frozen.
- GARCH parameters are re-estimated every 21 trading days, not daily; forecasts within a block use fixed parameters.
- Squared daily returns are an unbiased but very noisy proxy for latent variance; QLIKE is robust to that noise for ranking (Patton, 2011) but absolute loss levels are not interpretable as forecast error vs the true variance. Intraday realised variance would be a better proxy.
- Many pairwise tests are run; use the Holm-adjusted p-values, and treat unadjusted p-values as exploratory.
- With ~100+ monthly forecasts, power to detect small accuracy differences is low; a non-significant DM result is not evidence of equal accuracy.
- Residual diagnostics are in-sample and use full-sample fits.
- Nothing here is investment advice.

## 8. Reproducibility

Environment: Python 3.14.2. Command: `python -m src.run_all --data nifty --out results`. Seed: 0.
Install with `pip install -r requirements.txt`, then `python -m src.run_all --data nifty --out results`. See README for details.

Figures are in `results/figures/`.