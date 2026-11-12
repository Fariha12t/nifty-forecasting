"""End-to-end pipeline.

    python -m src.run_all --data nifty --out results          # real NIFTY 50 via Yahoo Finance
    python -m src.run_all --data nifty --csv nifty.csv        # your own Date,Close CSV
    python -m src.run_all --data synthetic --quick            # offline smoke test only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import mean_models as MM
from . import metrics as M
from . import plots as P
from . import stats_tests as ST
from . import volatility as V

BENCH = "Historical mean"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["nifty", "synthetic"], default="nifty")
    ap.add_argument("--csv", default=None, help="CSV with a date index and a Close column")
    ap.add_argument("--start", default="2007-09-17")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--init-months", type=int, default=96, help="initial training window for monthly models")
    ap.add_argument("--init-days", type=int, default=1500, help="initial training window for daily GARCH models")
    ap.add_argument("--refit-every", type=int, default=21, help="GARCH re-estimation frequency (trading days)")
    ap.add_argument("--max-origins", type=int, default=None, help="limit monthly forecast origins (most recent)")
    ap.add_argument("--max-vol-origins", type=int, default=None, help="limit daily forecast origins (most recent)")
    ap.add_argument("--quick", action="store_true", help="small run for smoke tests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.quick:
        args.max_origins = args.max_origins or 24
        args.max_vol_origins = args.max_vol_origins or 500
        args.refit_every = max(args.refit_every, 63)
        args.init_days = min(args.init_days, 1000)
    return args


def md(df: pd.DataFrame, floatfmt=".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def evaluate_mean(fc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bench = fc[fc["model"] == BENCH].set_index("date")
    rows, dm_rows = [], []
    for m in MM.MODELS:
        g = fc[fc["model"] == m].set_index("date")
        row = {"model": m}
        row.update(M.point_metrics(g["actual"], g["pred"], bench["pred"]))
        row.update(M.interval_metrics(g["actual"], g["lo"], g["hi"]))
        if m != BENCH:
            stat, p = M.dm_test((g["actual"] - g["pred"]) ** 2, (bench["actual"] - bench["pred"]) ** 2, h=1)
            row["DM stat vs hist. mean"], row["DM p"] = stat, p
        else:
            row["DM stat vs hist. mean"], row["DM p"] = np.nan, np.nan
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(dm_rows)


def evaluate_vol(vol: pd.DataFrame, proxy_col="actual"):
    vol = V.add_quantiles(vol.dropna(subset=["sigma2"]))
    vol["proxy"] = vol[proxy_col] ** 2  # squared return as the (noisy) variance proxy
    vol["qlike"] = M.qlike(vol["sigma2"], vol["proxy"])
    vol["se"] = (vol["sigma2"] - vol["proxy"]) ** 2
    # use common dates so every model is scored on the same days
    common = vol.groupby("date")["model"].nunique()
    common = common[common == vol["model"].nunique()].index
    vol = vol[vol["date"].isin(common)]
    ref = vol[vol["model"] == "EWMA (0.94)"].set_index("date")
    loss_rows, var_rows = [], []
    nlag = int(len(common) ** (1 / 3))
    for m, g in vol.groupby("model", sort=False):
        g = g.set_index("date")
        row = {"model": m, "QLIKE": g["qlike"].mean(), "MSE (var)": g["se"].mean()}
        if m != "EWMA (0.94)":
            stat, p = M.dm_test(g["qlike"], ref["qlike"], max_lag=nlag)
            row["DM stat vs EWMA"], row["DM p (QLIKE)"] = stat, p
        else:
            row["DM stat vs EWMA"], row["DM p (QLIKE)"] = np.nan, np.nan
        cover = ((g["actual"] >= g["q0.025"]) & (g["actual"] <= g["q0.975"])).mean()
        row["95% interval coverage"] = cover
        loss_rows.append(row)
        for level, col in ((0.01, "q0.01"), (0.05, "q0.05")):
            bt = M.var_backtest(g["actual"], g[col], level)
            var_rows.append({"model": m, "VaR level": f"{int((1-level)*100)}%", **bt})
    return vol, pd.DataFrame(loss_rows), pd.DataFrame(var_rows)


def build_report(ctx: dict) -> str:
    mm, lm, vr = ctx["mean_metrics"], ctx["vol_loss"], ctx["var_bt"]
    nonbench = mm[mm["model"] != BENCH]
    best_rmse = mm.sort_values("RMSE").iloc[0]
    sig = nonbench[nonbench["DM p"] < 0.05]
    if len(sig):
        mean_msg = "Models with a significantly different squared error from the historical mean (DM p < 0.05): " + ", ".join(
            f"{r.model} (OOS R² {r['OOS R2 vs hist. mean']:+.3f})" for _, r in sig.iterrows()
        ) + "."
    else:
        mean_msg = (
            "No model has a significantly different squared error from the historical-mean benchmark "
            "(all Diebold-Mariano p-values ≥ 0.05)."
        )
    best_q = lm.sort_values("QLIKE").iloc[0]
    pass1 = vr[(vr["VaR level"] == "99%") & (vr["Kupiec p"] > 0.05)]["model"].tolist()
    pass5 = vr[(vr["VaR level"] == "95%") & (vr["Kupiec p"] > 0.05)]["model"].tolist()
    sel = ctx["spec"]
    lines = [
        "# Results report (auto-generated)",
        "",
        f"Data: **{ctx['data_label']}**, {ctx['first']} to {ctx['last']}. "
        f"Monthly forecast origins: {ctx['n_mean_origins']} (expanding window, first training window {ctx['init_months']} months). "
        f"Daily volatility origins: {ctx['n_vol_origins']} (GARCH parameters re-estimated every {ctx['refit_every']} trading days).",
        "",
        "## 1. Stationarity and descriptive diagnostics",
        "",
        md(ctx["stationarity"]),
        "",
        "Daily return descriptives:",
        "",
        md(ctx["descr"].reset_index().rename(columns={"index": "statistic"}), floatfmt=".4g"),
        "",
        "## 2. Monthly return forecasts (rolling-origin, one step ahead)",
        "",
        f"Selected on the first window: {sel['ARIMA']}; {sel['SARIMA']}; {sel['ETS']}. Fit failures (fell back to the historical mean): {sel['fit_failures']}.",
        "",
        md(mm),
        "",
        f"**Headline:** lowest RMSE is {best_rmse['model']} ({best_rmse['RMSE']:.3f}). {mean_msg} "
        "OOS R² is relative to the expanding historical-mean forecast; negative means worse than that benchmark.",
        "",
        "Residual diagnostics (full-sample fits) Ljung-Box p-values: "
        + ", ".join(f"{k}: {v:.3f}" for k, v in ctx["resid_lb"].items()) + ".",
        "",
        "## 3. Volatility forecasts (rolling-origin, one step ahead)",
        "",
        md(ctx["garch_params"], floatfmt=".4g"),
        "",
        md(lm),
        "",
        f"**Headline:** lowest QLIKE is {best_q['model']} ({best_q['QLIKE']:.4f}). Lower QLIKE is better; DM tests compare QLIKE with the EWMA benchmark.",
        "",
        "## 4. Value-at-Risk backtests",
        "",
        md(vr),
        "",
        f"Models not rejected by the Kupiec test at the 5% level: 99% VaR: {', '.join(pass1) or 'none'}; 95% VaR: {', '.join(pass5) or 'none'}.",
        "",
        "## Caveats",
        "",
        "- ARIMA/SARIMA/ETS specifications are chosen by AIC on the first training window only; parameters are re-estimated at every origin.",
        "- Squared returns are a noisy variance proxy; QLIKE is used because it ranks forecasts consistently despite that noise.",
        "- Results describe one index over one sample period; they are not trading advice.",
        "",
        "Figures are in `results/figures/`.",
    ]
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig = out / "figures"

    print("[1/6] Loading data")
    if args.data == "synthetic":
        close = D.make_synthetic(seed=args.seed)
        label = "SYNTHETIC (pipeline test only, not NIFTY 50)"
    else:
        close = D.load_nifty(args.start, args.end, args.csv)
        label = "NIFTY 50 (^NSEI)" if not args.csv else f"NIFTY 50 from {args.csv}"
    ret = D.daily_returns(close)
    lp = D.monthly_log_price(close)
    r_m = D.monthly_returns(lp)
    print(f"      {len(close)} daily prices, {len(r_m)} monthly returns, {close.index[0].date()} to {close.index[-1].date()}")

    print("[2/6] Stationarity and descriptive tests")
    stat_tab = ST.stationarity_table(
        {"Log price (daily)": np.log(close), "Daily log returns": ret, "Monthly log price": lp / 100, "Monthly log returns": r_m}
    )
    stat_tab.round(4).to_csv(out / "stationarity_tests.csv", index=False)
    descr = ST.descriptives(ret)
    descr.to_csv(out / "return_descriptives.csv")
    print(stat_tab[["series", "ADF p", "KPSS p", "verdict"]].to_string(index=False))
    P.plot_overview(close, ret, fig / "overview.png")
    P.plot_acf_pacf(ret, fig / "acf_pacf_daily_returns.png")

    print("[3/6] Monthly mean models: rolling-origin cross-validation")
    fc, spec = MM.rolling_origin(lp, r_m, init=args.init_months, max_origins=args.max_origins)
    fc.to_csv(out / "monthly_forecasts.csv", index=False)
    mean_metrics, _ = evaluate_mean(fc)
    mean_metrics.round(4).to_csv(out / "monthly_metrics.csv", index=False)
    print(mean_metrics[["model", "RMSE", "OOS R2 vs hist. mean", "95% coverage", "DM p"]].round(4).to_string(index=False))
    best = mean_metrics.sort_values("RMSE").iloc[0]["model"]
    P.plot_forecast_intervals(fc, best, fig / "monthly_forecast_intervals_best.png")
    P.plot_cum_sse_diff(fc, BENCH, fig / "monthly_cum_sse_vs_benchmark.png")
    P.plot_metric_bars(mean_metrics, fig / "monthly_rmse.png")

    resid_lb = {}
    for name in ("ARIMA", "SARIMA"):
        resid, mdf = MM.final_arima_residuals(r_m, spec["_orders"][name])
        resid_lb[name] = ST.ljung_box_p(resid, lags=12, model_df=mdf)
        P.plot_residual_diagnostics(resid, f"{name} residual diagnostics (full sample)", fig / f"residuals_{name.lower()}.png")

    print("[4/6] Daily volatility models: rolling-origin forecasts")
    vol_parts = [
        V.rolling_garch(ret, name, args.init_days, args.refit_every, args.max_vol_origins) for name in V.SPECS
    ]
    garch_fc = pd.concat(vol_parts, ignore_index=True)
    dates = pd.DatetimeIndex(sorted(garch_fc["date"].unique()))
    vol_all = pd.concat([garch_fc, V.baseline_forecasts(ret, dates)], ignore_index=True)
    print("[5/6] Evaluating volatility forecasts, intervals and VaR")
    vol, vol_loss, var_bt = evaluate_vol(vol_all)
    vol.to_csv(out / "volatility_forecasts.csv", index=False)
    vol_loss.round(5).to_csv(out / "volatility_losses.csv", index=False)
    var_bt.round(5).to_csv(out / "var_backtests.csv", index=False)
    print(vol_loss.round(4).to_string(index=False))
    print(var_bt.round(4).to_string(index=False))

    params, results = V.fit_full_sample(ret)
    params.round(5).to_csv(out / "garch_full_sample_params.csv", index=False)
    best_garch = params.sort_values("BIC").iloc[0]["model"]
    std_resid = np.asarray(results[best_garch].std_resid)
    std_resid = std_resid[~np.isnan(std_resid)]
    P.plot_residual_diagnostics(std_resid, f"{best_garch}: standardised residuals (full sample)", fig / "residuals_garch.png")
    sq_lb = ST.ljung_box_p(std_resid**2, lags=10)
    resid_lb[f"{best_garch} (squared std. residuals)"] = sq_lb
    P.plot_vol_forecasts(vol, [m for m in V.SPECS] + ["EWMA (0.94)"], fig / "volatility_forecasts.png")
    best_var_model = vol_loss.sort_values("QLIKE").iloc[0]["model"]
    P.plot_var(vol, best_var_model, fig / "var_violations_best.png")

    print("[6/6] Writing report")
    ctx = dict(
        data_label=label, first=close.index[0].date(), last=close.index[-1].date(),
        n_mean_origins=int((fc["model"] == BENCH).sum()), n_vol_origins=int(vol["date"].nunique()),
        init_months=args.init_months, refit_every=args.refit_every, stationarity=stat_tab.round(4),
        descr=descr, mean_metrics=mean_metrics, spec=spec, resid_lb=resid_lb,
        garch_params=params, vol_loss=vol_loss, var_bt=var_bt,
    )
    (out / "REPORT.md").write_text(build_report(ctx))
    summary = {"data": label, "best_mean_model_by_rmse": best, "best_vol_model_by_qlike": best_var_model,
               "selected_specs": {k: v for k, v in spec.items() if not k.startswith("_")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Done. Outputs in {out.resolve()}")
    return ctx


if __name__ == "__main__":
    main()
