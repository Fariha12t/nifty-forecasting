from src.run_all import main


def test_pipeline_runs_end_to_end(tmp_path):
    main(["--data", "synthetic", "--quick", "--out", str(tmp_path)])
    for name in ("REPORT.md", "monthly_metrics.csv", "volatility_losses.csv", "var_backtests.csv", "stationarity_tests.csv"):
        assert (tmp_path / name).exists(), name
    assert len(list((tmp_path / "figures").glob("*.png"))) >= 10
    report = (tmp_path / "REPORT.md").read_text()
    assert "SYNTHETIC" in report  # synthetic runs must be labelled as such
    assert "{'ARIMA': 0, 'SARIMA': 0, 'ETS': 0}" in report  # no silent fallbacks
