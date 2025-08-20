import multiobjective.experiment as experiment
from multiobjective.config import Config, NSGAConfig, PSOConfig, GWOConfig
from multiobjective.algorithms.greedy import greedy_run
import multiobjective.algorithms as algorithms


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

    assert {"series", "indicators", "scs", "meta"} <= set(result.keys())
    assert "greedy" in result["series"]
