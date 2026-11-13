"""End-to-end pipeline.

    python -m src.run_all --data nifty --out results          # real NIFTY 50 via Yahoo Finance
    python -m src.run_all --data nifty --csv data/nifty.csv   # your own Date,Close CSV
    python -m src.run_all --data synthetic --quick            # offline smoke test only (never report these numbers)
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from . import data as D
from . import mean_models as MM
from . import metrics as M
from . import plots as P
from . import stats_tests as ST
from . import volatility as V

BENCH = "Historical mean"
VOL_REF = "EWMA (0.94)"
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


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
    ap.add_argument("--no-readme", action="store_true", help="do not update the README results block")
    args = ap.parse_args(argv)
    if args.quick:
        args.max_origins = args.max_origins or 24
        args.max_vol_origins = args.max_vol_origins or 500
        args.refit_every = max(args.refit_every, 63)
        args.init_days = min(args.init_days, 1000)
    return args


def md(df: pd.DataFrame, floatfmt=".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate_mean(fc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics per model and pairwise Diebold-Mariano tests (squared error; Newey-West + HLN correction)."""
    bench = fc[fc["model"] == BENCH].set_index("date")
    wide = {m: fc[fc["model"] == m].set_index("date") for m in MM.MODELS}
    rows = []
    for m in MM.MODELS:
        g = wide[m]
        row = {"model": m, "n_forecasts": len(g)}
        row.update(M.point_metrics(g["actual"], g["pred"], bench["pred"]))
        row.update(M.interval_metrics(g["actual"], g["lo"], g["hi"]))
        if m != BENCH:
            stat, p = M.dm_test((g["actual"] - g["pred"]) ** 2, (bench["actual"] - bench["pred"]) ** 2, max_lag=M.nw_lag(len(g)))
            row["DM stat vs hist. mean"], row["DM p vs hist. mean"] = stat, p
        else:
            row["DM stat vs hist. mean"], row["DM p vs hist. mean"] = np.nan, np.nan
        rows.append(row)
    dm = []
    for a, b in itertools.combinations(MM.MODELS, 2):
        ea, eb = (wide[a]["actual"] - wide[a]["pred"]) ** 2, (wide[b]["actual"] - wide[b]["pred"]) ** 2
        stat, p = M.dm_test(ea, eb, max_lag=M.nw_lag(len(ea)))
        dm.append({"family": "DM monthly (squared error)", "model_a": a, "model_b": b, "statistic": stat, "p_value": p,
                   "note": "negative = A has lower loss", "mean_loss_a": float(ea.mean()), "mean_loss_b": float(eb.mean())})
    dm = pd.DataFrame(dm)
    dm["p_holm"] = multipletests(dm["p_value"], method="holm")[1]
    return pd.DataFrame(rows), dm


def evaluate_vol(vol: pd.DataFrame):
    """QLIKE/MSE/MAE, pairwise DM on QLIKE and 95%/99% VaR backtests on common dates for every model."""
    vol = V.add_quantiles(vol.dropna(subset=["sigma2"]))
    vol["proxy"] = vol["actual"] ** 2  # squared return as the (noisy) variance proxy
    vol["qlike"] = M.qlike(vol["sigma2"], vol["proxy"])
    n_models = vol["model"].nunique()
    common = vol.groupby("date")["model"].nunique()
    common = common[common == n_models].index
    vol = vol[vol["date"].isin(common)].copy()
    ql = {m: g.set_index("date")["qlike"] for m, g in vol.groupby("model", sort=False)}
    nlag = M.nw_lag(len(common))
    loss_rows, var_rows = [], []
    for m, g in vol.groupby("model", sort=False):
        g = g.set_index("date")
        row = {"model": m, "n_forecasts": len(g), **M.vol_point_metrics(g["sigma2"], g["proxy"])}
        row["Mean forecast vol (%/day)"] = float(np.sqrt(g["sigma2"]).mean())
        row["95% return-interval coverage"] = float(((g["actual"] >= g["q0.025"]) & (g["actual"] <= g["q0.975"])).mean())
        loss_rows.append(row)
        for level, col in ((0.01, "q0.01"), (0.05, "q0.05")):
            bt = M.var_backtest(g["actual"], g[col], level)
            lr_pof, _ = M.kupiec_pof((g["actual"] < g[col]).to_numpy(), level)
            var_rows.append({"model": m, "VaR level": f"{int((1 - level) * 100)}%", "n": len(g),
                             "Exceedances": int((g["actual"] < g[col]).sum()), **bt})
    dm = []
    for a, b in itertools.combinations(list(ql), 2):
        stat, p = M.dm_test(ql[a], ql[b], max_lag=nlag)
        dm.append({"family": "DM volatility (QLIKE)", "model_a": a, "model_b": b, "statistic": stat, "p_value": p,
                   "note": "negative = A has lower QLIKE", "mean_loss_a": float(ql[a].mean()), "mean_loss_b": float(ql[b].mean())})
    dm = pd.DataFrame(dm)
    dm["p_holm"] = multipletests(dm["p_value"], method="holm")[1]
    return vol, pd.DataFrame(loss_rows), pd.DataFrame(var_rows), dm


def diagnostics_rows(ret, r_m, stat_tab, descr_d, descr_m) -> list[dict]:
    rows = []
    for _, r in stat_tab.iterrows():
        rows.append({"category": "stationarity", "test": "ADF (H0: unit root)", "series": r["series"], "statistic": r["ADF stat"], "p_value": r["ADF p"], "detail": r["verdict"]})
        rows.append({"category": "stationarity", "test": "KPSS (H0: stationary)", "series": r["series"], "statistic": r["KPSS stat"], "p_value": r["KPSS p"], "detail": r["verdict"]})
    for label, d in (("daily returns", descr_d), ("monthly returns", descr_m)):
        for k, v in d["value"].items():
            if "p" in k.split()[-1] or k.endswith(" p") or ") p" in k or "Jarque" in k:
                rows.append({"category": "distribution/dependence", "test": k, "series": label, "statistic": np.nan, "p_value": v, "detail": ""})
            else:
                rows.append({"category": "descriptive", "test": k, "series": label, "statistic": v, "p_value": np.nan, "detail": ""})
    for r in ST.clustering_rows(ret):
        rows.append({"category": "volatility clustering", **r, "detail": ""})
    return rows


def verdict_sentences(stat_tab, clustering) -> list[str]:
    out = []
    for _, r in stat_tab.iterrows():
        out.append(f"- {r['series']}: ADF p={r['ADF p']:.4f}, KPSS p={r['KPSS p']:.4f} → **{r['verdict']}**.")
    lb2 = [c for c in clustering if c["series"] == "daily squared returns"]
    lm = [c for c in clustering if c["test"].startswith("ARCH-LM")]
    if lb2 and lm:
        if max(c["p_value"] for c in lb2 + lm) < 0.05:
            out.append("- Volatility clustering: every Ljung-Box (squared returns) and ARCH-LM test rejects the null at 5%, so the data show statistically significant ARCH effects.")
        else:
            out.append("- Volatility clustering: at least one Ljung-Box/ARCH-LM test does not reject at 5%; the evidence for ARCH effects is mixed. See `statistical_tests.csv`.")
    return out


def build_report(ctx: dict) -> str:
    mm, lm, vr = ctx["mean_metrics"], ctx["vol_loss"], ctx["var_bt"]
    nonbench = mm[mm["model"] != BENCH]
    best_rmse = mm.sort_values("RMSE").iloc[0]
    sig = nonbench[nonbench["DM p vs hist. mean"] < 0.05]
    dmm = ctx["dm_monthly"]
    sig_holm = dmm[dmm["p_holm"] < 0.05]
    if len(sig):
        mean_msg = "Models whose squared error differs significantly from the historical mean (DM p < 0.05, unadjusted): " + ", ".join(
            f"{r.model} (OOS R² {r['OOS R2 vs hist. mean']:+.3f}, DM stat {r['DM stat vs hist. mean']:+.2f})" for _, r in sig.iterrows()) + "."
    else:
        mean_msg = "No model's squared error is significantly different from the historical-mean benchmark (all DM p ≥ 0.05)."
    holm_msg = (f"After Holm adjustment across all {len(dmm)} pairwise comparisons, {len(sig_holm)} pair(s) remain significant at 5%."
                if len(dmm) else "")
    best_q = lm.sort_values("QLIKE").iloc[0]
    dmv = ctx["dm_vol"]
    pass1 = vr[(vr["VaR level"] == "99%") & (vr["Cond. coverage p"] > 0.05)]["model"].tolist()
    pass5 = vr[(vr["VaR level"] == "95%") & (vr["Cond. coverage p"] > 0.05)]["model"].tolist()
    sel = ctx["spec"]
    ev = ctx["fallback_events"]
    ev_txt = "none" if ev.empty else ", ".join(f"{m}/{e}: {n}" for (m, e), n in ev.groupby(["model", "event"]).size().items())
    synth = "SYNTHETIC" in ctx["data_label"]
    lines = [
        "# Results report (auto-generated)",
        "",
        "> **WARNING: these results come from SYNTHETIC data and are a pipeline test only. They say nothing about NIFTY 50.**" if synth else "",
        "",
        "## 1. Dataset and methodology",
        "",
        f"- Source: **{ctx['data_label']}**; downloaded/loaded on {ctx['validation'].get('download_date_utc', 'n/a')}.",
        f"- Daily sample: {ctx['first']} to {ctx['last']}, {ctx['n_daily']} closing prices, {ctx['n_daily'] - 1} daily log returns (percent).",
        f"- Monthly sample: {ctx['n_monthly']} month-end returns (partial trailing month dropped).",
        f"- Data validation: {json.dumps({k: v for k, v in ctx['validation'].items() if k.startswith('n_') or k.startswith('max') or k.startswith('is_')})}.",
        "- Cleaning: NaN closes and duplicate dates dropped, sorted chronologically; non-trading days are not filled; non-positive prices raise an error.",
        "- Log returns: r_t = 100·(ln P_t − ln P_{t−1}); monthly returns use month-end closes.",
        "",
        "## 2. Forecasting design",
        "",
        f"- Monthly: expanding window, one-step ahead, first training window {ctx['init_months']} months, **{ctx['n_mean_origins']} out-of-sample forecasts**, models re-estimated at every origin.",
        f"- Selected on the first window only: {sel['ARIMA']}; {sel['SARIMA']}; {sel['ETS']}.",
        f"- Daily: GARCH-family models (Student-t, constant mean) re-estimated every {ctx['refit_every']} trading days on data strictly before each block; **{ctx['n_vol_origins']} out-of-sample daily forecasts** on common dates; EWMA λ=0.94 and rolling-250d baselines.",
        "- Baseline VaR quantiles use a normal distribution (EWMA and rolling window have no fitted tail parameter); GARCH-family VaR uses the fitted standardised Student-t. Differences in VaR performance therefore mix variance forecasts and tail assumptions.",
        "",
        "## 3. Diagnostics",
        "",
        *verdict_sentences(ctx["stationarity"], ctx["clustering"]),
        "",
        md(ctx["stationarity"]),
        "",
        "Daily return descriptives:",
        "",
        md(ctx["descr"].reset_index().rename(columns={"index": "statistic"}), floatfmt=".4g"),
        "",
        "Residual diagnostics (full-sample fits, in-sample; not forecast evaluation):",
        "",
        md(ctx["resid_tab"], floatfmt=".4g"),
        "",
        "## 4. Monthly return forecasts",
        "",
        md(mm),
        "",
        f"**Headline:** lowest RMSE is {best_rmse['model']} ({best_rmse['RMSE']:.4f}). {mean_msg} {holm_msg} "
        "OOS R² is relative to the expanding historical-mean forecast (negative = worse than that benchmark). "
        "Directional accuracy is undefined for zero drift (it never predicts a sign). "
        "A statistically significant DM result is not by itself economically meaningful: compare the size of the RMSE/MAE gap to the monthly return scale.",
        "",
        f"Fit failures (fell back to the historical mean forecast): {sel['fit_failures']}. Statsmodels warnings counted: {sel.get('fit_warnings')}.",
        "",
        "## 5. Daily volatility forecasts",
        "",
        md(ctx["garch_params"], floatfmt=".4g"),
        "",
        md(lm),
        "",
        f"**Headline:** lowest QLIKE is {best_q['model']} ({best_q['QLIKE']:.4f}). Lower is better. "
        f"Pairwise DM on QLIKE: {int((dmv['p_value'] < 0.05).sum())} of {len(dmv)} pairs significant at 5% unadjusted, {int((dmv['p_holm'] < 0.05).sum())} after Holm adjustment.",
        "",
        f"GARCH fallback/failure events: {ev_txt}. Full log: `fallback_events.csv`.",
        "",
        "## 6. Value-at-Risk backtests",
        "",
        md(vr),
        "",
        f"Models not rejected by the conditional-coverage test at 5%: 99% VaR: {', '.join(pass1) or 'none'}; 95% VaR: {', '.join(pass5) or 'none'}. "
        "Kupiec tests unconditional coverage, Christoffersen tests independence of exceedances, and conditional coverage combines them (χ²₂).",
        "",
        "## 7. Limitations",
        "",
        "- Single index, single sample period; conclusions may not transfer to other markets or regimes.",
        "- ARIMA/SARIMA/ETS specifications are selected once by AIC on the first training window and then frozen.",
        "- GARCH parameters are re-estimated every 21 trading days, not daily; forecasts within a block use fixed parameters.",
        "- Squared daily returns are an unbiased but very noisy proxy for latent variance; QLIKE is robust to that noise for ranking (Patton, 2011) but absolute loss levels are not interpretable as forecast error vs the true variance. Intraday realised variance would be a better proxy.",
        "- Many pairwise tests are run; use the Holm-adjusted p-values, and treat unadjusted p-values as exploratory.",
        "- With ~100+ monthly forecasts, power to detect small accuracy differences is low; a non-significant DM result is not evidence of equal accuracy.",
        "- Residual diagnostics are in-sample and use full-sample fits.",
        "- Nothing here is investment advice.",
        "",
        "## 8. Reproducibility",
        "",
        f"Environment: Python {platform.python_version()}. Command: `{ctx['command']}`. Seed: {ctx['seed']}.",
        "Install with `pip install -r requirements.txt`, then `python -m src.run_all --data nifty --out results`. See README for details.",
        "",
        "Figures are in `results/figures/`.",
    ]
    return "\n".join(lines)


def update_readme(readme: Path, ctx: dict) -> None:
    """Fill the README results block from real-data outputs (never from synthetic runs)."""
    if not readme.exists():
        return
    mm, lm, vr = ctx["mean_metrics"], ctx["vol_loss"], ctx["var_bt"]
    keep = ["model", "RMSE", "MAE", "OOS R2 vs hist. mean", "Directional acc.", "95% coverage", "DM p vs hist. mean"]
    block = [
        README_START,
        f"_Auto-generated by `src.run_all` from {ctx['data_label']}, {ctx['first']} to {ctx['last']}: "
        f"{ctx['n_daily']} daily prices, {ctx['n_monthly']} monthly returns, {ctx['n_mean_origins']} monthly and {ctx['n_vol_origins']} daily out-of-sample forecasts._",
        "",
        "**Monthly return forecasts (one step ahead, expanding window)**",
        "",
        md(mm[keep]),
        "",
        "**Daily volatility forecasts**",
        "",
        md(lm[["model", "QLIKE", "MSE (variance)", "MAE (sigma vs |r|)"]]),
        "",
        "**VaR backtests (p-values)**",
        "",
        md(vr[["model", "VaR level", "Exceedances", "Violation rate", "Kupiec p", "Independence p", "Cond. coverage p"]]),
        "",
        "Full tables, tests, and interpretation: [`results/REPORT.md`](results/REPORT.md).",
        README_END,
    ]
    text = readme.read_text()
    if README_START in text and README_END in text:
        a, b = text.index(README_START), text.index(README_END) + len(README_END)
        text = text[:a] + "\n".join(block) + text[b:]
        readme.write_text(text)


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig = out / "figures"

    print("[1/6] Loading and validating data")
    if args.data == "synthetic":
        close = D.make_synthetic(seed=args.seed)
        label = "SYNTHETIC (pipeline test only, not NIFTY 50)"
        validation = {**D.validate_prices(close), "source": "synthetic", "n_clean": len(close),
                      "first_date": str(close.index[0].date()), "last_date": str(close.index[-1].date())}
    else:
        close, validation = D.load_nifty(args.start, args.end, args.csv, return_report=True)
        label = "NIFTY 50 (^NSEI, Yahoo Finance)" if not args.csv else f"NIFTY 50 from {args.csv}"
        close.to_frame().to_csv(out / "nifty_close_used.csv", index_label="Date")  # exact series the results are computed from
    (out / "data_validation.json").write_text(json.dumps(validation, indent=2))
    ret = D.daily_returns(close)
    lp = D.monthly_log_price(close)
    r_m = D.monthly_returns(lp)
    print(f"      {len(close)} daily prices, {len(r_m)} monthly returns, {close.index[0].date()} to {close.index[-1].date()}")

    print("[2/6] Stationarity, distribution and clustering diagnostics")
    stat_tab = ST.stationarity_table(
        {"Log price (daily)": np.log(close), "Daily log returns": ret, "Monthly log price": lp / 100, "Monthly log returns": r_m}
    )
    stat_tab.round(4).to_csv(out / "stationarity_tests.csv", index=False)
    descr, descr_m = ST.descriptives(ret), ST.descriptives(r_m, lags=min(10, len(r_m) // 5))
    descr.to_csv(out / "return_descriptives.csv")
    clustering = ST.clustering_rows(ret)
    P.plot_overview(close, ret, fig / "01_price_returns_volatility.png")
    P.plot_monthly_returns(r_m, fig / "02_monthly_returns.png")
    P.plot_return_distribution(ret, fig / "03_return_distribution_qq.png")
    P.plot_acf_pacf(ret, fig / "04_acf_pacf_daily_returns.png")
    P.plot_monthly_acf_pacf(r_m, fig / "05_acf_pacf_monthly_returns.png")
    P.plot_volatility_clustering(ret, fig / "06_volatility_clustering.png")

    print("[3/6] Monthly mean models: rolling-origin forecasts")
    fc, spec = MM.rolling_origin(lp, r_m, init=args.init_months, max_origins=args.max_origins)
    fc.to_csv(out / "monthly_forecasts.csv", index=False)
    mean_metrics, dm_monthly = evaluate_mean(fc)
    mean_metrics.round(6).to_csv(out / "monthly_forecast_metrics.csv", index=False)
    print(mean_metrics[["model", "RMSE", "OOS R2 vs hist. mean", "95% coverage", "DM p vs hist. mean"]].round(4).to_string(index=False))
    best = mean_metrics.sort_values("RMSE").iloc[0]["model"]
    P.plot_forecast_comparison(fc, fig / "07_monthly_actual_vs_forecast.png")
    P.plot_forecast_intervals(fc, best, fig / "08_monthly_intervals_best.png")
    P.plot_interval_comparison(fc, fig / "09_monthly_intervals_all_models.png")
    P.plot_cum_sse_diff(fc, BENCH, fig / "10_monthly_cum_sse_vs_benchmark.png")
    P.plot_metric_bars(mean_metrics, fig / "11_monthly_rmse.png")

    resid_rows, resid_lb = [], {}
    for name in ("ARIMA", "SARIMA"):
        resid, mdf = MM.final_arima_residuals(r_m, spec["_orders"][name])
        resid_lb[name] = ST.ljung_box_p(resid, lags=12, model_df=mdf)
        resid_rows += [{**r, "series": f"{name} residuals"} for r in ST.ljung_box_table(resid, name, lags=(6, 12, 24), model_df=mdf)]
        st, p = ST.arch_lm_p(resid, min(6, len(resid) // 10))
        resid_rows.append({"test": "ARCH-LM(6)", "series": f"{name} residuals", "statistic": st, "p_value": p})
        P.plot_residual_diagnostics(resid, f"{name} residual diagnostics (full sample, in-sample)", fig / f"12_residuals_{name.lower()}.png")

    print("[4/6] Daily volatility models: rolling-origin forecasts")
    parts, events = [], []
    for name in V.SPECS:
        f, ev = V.rolling_garch(ret, name, args.init_days, args.refit_every, args.max_vol_origins)
        parts.append(f)
        events += ev
    garch_fc = pd.concat(parts, ignore_index=True)
    fallback_events = pd.DataFrame(events, columns=["model", "origin_date", "event", "detail"])
    fallback_events.to_csv(out / "fallback_events.csv", index=False)
    dates = pd.DatetimeIndex(sorted(garch_fc["date"].unique()))
    vol_all = pd.concat([garch_fc, V.baseline_forecasts(ret, dates)], ignore_index=True)

    print("[5/6] Evaluating volatility forecasts and VaR")
    vol, vol_loss, var_bt, dm_vol = evaluate_vol(vol_all)
    vol.to_csv(out / "volatility_forecasts.csv", index=False)
    vol_loss.round(6).to_csv(out / "daily_volatility_metrics.csv", index=False)
    var_bt.round(6).to_csv(out / "var_backtests.csv", index=False)
    print(vol_loss.round(4).to_string(index=False))
    print(var_bt.round(4).to_string(index=False))

    params, results = V.fit_full_sample(ret)
    params.round(5).to_csv(out / "garch_full_sample_params.csv", index=False)
    best_garch = params.sort_values("BIC").iloc[0]["model"]
    std_resid = np.asarray(results[best_garch].std_resid)
    std_resid = std_resid[~np.isnan(std_resid)]
    resid_rows += [{**r, "series": f"{best_garch} std. residuals"} for r in ST.ljung_box_table(std_resid, best_garch, lags=(5, 10, 20))]
    resid_rows += [{**r, "series": f"{best_garch} squared std. residuals"} for r in ST.ljung_box_table(std_resid**2, best_garch, lags=(5, 10, 20))]
    st, p = ST.arch_lm_p(std_resid, 10)
    resid_rows.append({"test": "ARCH-LM(10)", "series": f"{best_garch} std. residuals", "statistic": st, "p_value": p})
    resid_tab = pd.DataFrame(resid_rows)[["series", "test", "statistic", "p_value"]]
    P.plot_residual_diagnostics(std_resid, f"{best_garch}: standardised residuals (full sample, in-sample)", fig / "13_residuals_garch.png")
    P.plot_conditional_vol(ret, results, fig / "14_garch_conditional_volatility.png")
    P.plot_vol_forecasts(vol, list(V.SPECS) + [VOL_REF], fig / "15_volatility_forecasts.png")
    best_var_model = vol_loss.sort_values("QLIKE").iloc[0]["model"]
    P.plot_var(vol, best_var_model, fig / "16_var99_exceedances_best.png", level=0.01)
    P.plot_var(vol, best_var_model, fig / "17_var95_exceedances_best.png", level=0.05)

    print("[6/6] Writing statistical tests, report")
    tests = pd.DataFrame(diagnostics_rows(ret, r_m, stat_tab, descr, descr_m))
    resid_long = resid_tab.assign(category="residual diagnostics", detail="in-sample, full-sample fit")
    dm_all = pd.concat([dm_monthly, dm_vol], ignore_index=True).rename(columns={"family": "category"})
    dm_all["series"] = dm_all["model_a"] + " vs " + dm_all["model_b"]
    dm_all["test"] = "Diebold-Mariano (Newey-West, HLN)"
    dm_all["detail"] = dm_all["note"] + "; Holm-adjusted p=" + dm_all["p_holm"].map("{:.4g}".format)
    cols = ["category", "test", "series", "statistic", "p_value", "detail"]
    stat_all = pd.concat([tests[cols], resid_long[cols], dm_all[cols]], ignore_index=True)
    stat_all.to_csv(out / "statistical_tests.csv", index=False)
    dm_all.drop(columns=["category", "series", "test", "detail"]).assign(
        family=dm_all["category"]).to_csv(out / "dm_pairwise_detail.csv", index=False)

    ctx = dict(
        data_label=label, first=close.index[0].date(), last=close.index[-1].date(), validation=validation,
        n_daily=len(close), n_monthly=len(r_m), n_mean_origins=int((fc["model"] == BENCH).sum()),
        n_vol_origins=int(vol["date"].nunique()), init_months=args.init_months, refit_every=args.refit_every,
        stationarity=stat_tab.round(4), descr=descr, mean_metrics=mean_metrics, spec=spec, resid_tab=resid_tab,
        garch_params=params, vol_loss=vol_loss, var_bt=var_bt, dm_monthly=dm_monthly, dm_vol=dm_vol,
        clustering=clustering, fallback_events=fallback_events, seed=args.seed,
        command="python -m src.run_all " + " ".join(argv if argv is not None else __import__("sys").argv[1:]),
    )
    (out / "REPORT.md").write_text(build_report(ctx))
    summary = {"data": label, "best_mean_model_by_rmse": best, "best_vol_model_by_qlike": best_var_model,
               "selected_specs": {k: v for k, v in spec.items() if not k.startswith("_")},
               "n_garch_fallback_events": int(len(fallback_events))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if args.data == "nifty" and not args.no_readme:
        update_readme(Path("README.md"), ctx)
    print(f"Done. Outputs in {out.resolve()}")
    return ctx


if __name__ == "__main__":
    main()
