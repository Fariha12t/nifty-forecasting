import numpy as np

from src import metrics as M


def test_kupiec_perfect_rate_gives_p_one():
    viol = np.zeros(1000, dtype=int)
    viol[::100] = 1  # exactly 1% violations
    lr, p = M.kupiec_pof(viol, 0.01)
    assert lr < 1e-9 and p > 0.99


def test_kupiec_rejects_too_many_violations():
    viol = np.zeros(1000, dtype=int)
    viol[:60] = 1  # 6% against an expected 1%
    _, p = M.kupiec_pof(viol, 0.01)
    assert p < 0.001


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=int)
    clustered[100:130] = 1
    spread = np.zeros(1000, dtype=int)
    spread[::33] = 1
    assert M.christoffersen_ind(clustered)[1] < 0.01
    assert M.christoffersen_ind(spread)[1] > 0.05


def test_dm_identical_losses():
    x = np.random.default_rng(0).random(100)
    stat, p = M.dm_test(x, x)
    assert abs(stat) < 1e-9 and p > 0.99


def test_dm_detects_clear_winner():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.1, 300)
    b = rng.normal(2.0, 0.1, 300)
    stat, p = M.dm_test(a, b)
    assert stat < 0 and p < 0.001  # A has lower loss


def test_qlike_minimised_at_true_variance():
    rng = np.random.default_rng(2)
    true_var = 4.0
    proxy = true_var * rng.standard_normal(20000) ** 2
    at_truth = M.qlike(np.full_like(proxy, true_var), proxy).mean()
    assert at_truth < M.qlike(np.full_like(proxy, 2 * true_var), proxy).mean()
    assert at_truth < M.qlike(np.full_like(proxy, 0.5 * true_var), proxy).mean()


def test_interval_metrics_coverage():
    actual = np.array([0.0, 1.0, 2.0, 3.0])
    out = M.interval_metrics(actual, actual - 0.5, actual + 0.5)
    assert out["95% coverage"] == 1.0 and abs(out["Mean width"] - 1.0) < 1e-12
