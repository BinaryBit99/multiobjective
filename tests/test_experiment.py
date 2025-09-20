import multiobjective.experiment as experiment
from multiobjective.config import Config, NSGAConfig, PSOConfig, GWOConfig
from multiobjective.algorithms.greedy import greedy_run
import multiobjective.algorithms as algorithms
import pytest
from multiobjective.errors import CoverageError
from multiobjective.qos import reg_err


def test_run_experiment_minimal(monkeypatch):
    cfg = Config(
        num_times=1,
        num_services=4,
        coverage_fraction=0.1,
        nsga=NSGAConfig(population_size=4, max_generations=2),
        pso=PSOConfig(swarm_size=4, max_iterations=2),
        gwo=GWOConfig(wolf_size=4, max_iters=2),
    )

    monkeypatch.setattr(algorithms, "ALG_REGISTRY", {"greedy": greedy_run})

    class DummyStreak:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

    monkeypatch.setattr(experiment, "StreakTracker", DummyStreak)

    result = experiment.run_experiment(cfg)

    assert {"series", "indicators", "meta"} <= set(result.keys())
    assert "greedy" in result["series"]
    scs_section = result["series"]["greedy"].get("scs")
    assert scs_section is not None and set(scs_section.keys()) == {"tp", "res"}
    assert len(scs_section["tp"]["actual"]) == cfg.num_times
    assignments = result["series"]["greedy"]["assignments"]["tp"]
    assert len(assignments) == cfg.num_times
    assert len(assignments[0]) == result["meta"]["num_consumers"]

    times_section = result["series"]["greedy"].get("times")
    assert times_section is not None and set(times_section.keys()) == {"tp", "res"}
    assert len(times_section["tp"]) == cfg.num_times
    assert all(t2 >= t1 for t1, t2 in zip(times_section["tp"], times_section["tp"][1:]))


def test_run_experiment_uses_algorithm_assignments(monkeypatch):
    cfg = Config(
        num_times=2,
        num_services=4,
        coverage_fraction=1.0,
        nsga=NSGAConfig(population_size=2, max_generations=1),
        pso=PSOConfig(swarm_size=2, max_iterations=1),
        gwo=GWOConfig(wolf_size=2, max_iters=1),
    )

    def dummy_alg(cfg, rng_pool, records, cost_per, err_type, metrics, streaks, norm_fn):
        num_times = cfg.num_times
        num_consumers = cfg.num_consumers
        errs = [0.0] * num_times
        costs = [0.0] * num_times
        stds = [0.0] * num_times
        times = [0.0] * num_times
        assignments = [[0 for _ in range(num_consumers)] for _ in range(num_times)]
        scs_actual = [0.1 for _ in range(num_times)]
        scs_expected = [0.2 for _ in range(num_times)]
        for t in range(num_times):
            metrics.record("greedy", err_type, t, [(0.0, 0.0)])
        return errs, costs, stds, times, {
            "assignments": assignments,
            "scs_actual": scs_actual,
            "scs_expected": scs_expected,
        }

    monkeypatch.setattr(algorithms, "ALG_REGISTRY", {"greedy": dummy_alg})

    result = experiment.run_experiment(cfg)
    series = result["series"]["greedy"]
    assert series["scs"]["tp"]["actual"] == [0.1, 0.1]
    assert series["scs"]["tp"]["expected"] == [0.2, 0.2]
    expected_assignments = [[0] * cfg.num_consumers for _ in range(cfg.num_times)]
    assert series["assignments"]["tp"] == expected_assignments


def test_run_experiment_no_feasible_pairs():
    cfg = Config(
        num_times=1,
        num_services=4,
        coverage_fraction=0.0,
    )

    with pytest.raises(CoverageError):
        experiment.run_experiment(cfg)


def test_norm_err_identical_values(monkeypatch):
    cfg = Config(
        num_times=1,
        num_services=2,
        ratio_str="one_one",
        coverage_fraction=1.0,
        nsga=NSGAConfig(population_size=2, max_generations=1),
        pso=PSOConfig(swarm_size=2, max_iterations=1),
        gwo=GWOConfig(wolf_size=2, max_iters=1),
    )

    def dummy_alg(cfg, rng_pool, records, cost_per, err_type, metrics, norm_fn):
        errs, costs, stds, times = [], [], [], []
        for t in range(cfg.num_times):
            prods, cons = records[t]
            e = norm_fn(err_type, reg_err(prods[0], cons[0], err_type), t)
            c = norm_fn("__cost__", prods[0].cost, t)
            metrics.record("dummy", err_type, t, [(e, c)])
            errs.append(e)
            costs.append(c)
            stds.append(0.0)
            times.append(0.0)
        return errs, costs, stds, times

    monkeypatch.setattr(algorithms, "ALG_REGISTRY", {"dummy": dummy_alg})

    result = experiment.run_experiment(cfg)
    series = result["series"]["dummy"]
    for te in ["tp", "res"]:
        assert series["errors"][te] == [0.5]
        assert series["costs"][te] == [0.5]
