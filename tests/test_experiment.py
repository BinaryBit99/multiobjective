import multiobjective.experiment as experiment
from multiobjective.config import Config, NSGAConfig, PSOConfig, GWOConfig
from multiobjective.algorithms.greedy import greedy_run
import multiobjective.algorithms as algorithms
import pytest
from multiobjective.errors import CoverageError


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


def test_run_experiment_no_feasible_pairs():
    cfg = Config(
        num_times=1,
        num_services=4,
        coverage_fraction=0.0,
    )

    with pytest.raises(CoverageError):
        experiment.run_experiment(cfg)
