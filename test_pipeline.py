import pandas as pd

from src.run_all import main


def test_pipeline_runs_end_to_end(tmp_path):
    main(["--data", "synthetic", "--quick", "--out", str(tmp_path)])
    for name in ("REPORT.md", "monthly_forecast_metrics.csv", "monthly_forecasts.csv", "daily_volatility_metrics.csv",
                 "volatility_forecasts.csv", "var_backtests.csv", "statistical_tests.csv", "fallback_events.csv",
                 "data_validation.json"):
        assert (tmp_path / name).exists(), name
    assert len(list((tmp_path / "figures").glob("*.png"))) >= 15
    report = (tmp_path / "REPORT.md").read_text()
    assert "SYNTHETIC" in report  # synthetic runs must be labelled as such
    assert "'fit_failures'" not in report and "{'ARIMA': 0" in report or "fit_failures" in report or "Fit failures" in report
    st = pd.read_csv(tmp_path / "statistical_tests.csv")
    assert {"stationarity", "volatility clustering", "residual diagnostics"} <= set(st["category"])
    assert st["category"].str.contains("DM").any()
