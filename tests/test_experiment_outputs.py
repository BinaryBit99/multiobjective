import pytest

from multiobjective.config import Config
import multiobjective.experiment as experiment
from multiobjective.metrics.scs import SCSComponents
from multiobjective import algorithms


@pytest.fixture
def tiny_cfg() -> Config:
    """Small configuration with deterministic seed."""
    return Config(
        num_times=2,
        num_services=6,
        coverage_fraction=1.0,
        master_seed=0,
        scs_mc_rollouts=1,
    )


def test_run_experiment_outputs_structure(tiny_cfg, monkeypatch):
    """run_experiment should produce structured numeric outputs."""

    def dummy_alg(cfg, rng_pool, records, cost_per, err_type, metrics, streaks, norm_fn):
        series = [0.0] * cfg.num_times
        for t in range(cfg.num_times):
            metrics.record("dummy", err_type, t, [])
        return series, series, series

    monkeypatch.setattr(algorithms, "ALG_REGISTRY", {"dummy": dummy_alg})
    monkeypatch.setattr(experiment, "scs", lambda *a, **k: (0.0, SCSComponents()))
    monkeypatch.setattr(
        experiment,
        "expected_scs_next",
        lambda *a, **k: (0.0, SCSComponents()),
    )

    result = experiment.run_experiment(tiny_cfg)

    assert {"series", "indicators", "scs", "meta"} <= result.keys()
    assert set(result["scs"].keys()) == {"tp", "res", "E_tp", "E_res"}

    for values in result["scs"].values():
        assert len(values) == tiny_cfg.num_times
        assert all(isinstance(v, (int, float)) for v in values)

    for series in result["series"].values():
        for section in ("errors", "costs", "stds"):
            for values in series[section].values():
                assert len(values) == tiny_cfg.num_times
                assert all(isinstance(v, (int, float)) for v in values)

    for metrics in result["indicators"].values():
        for by_err in metrics.values():
            for values in by_err.values():
                assert len(values) == tiny_cfg.num_times
                assert all(isinstance(v, (int, float)) for v in values)
