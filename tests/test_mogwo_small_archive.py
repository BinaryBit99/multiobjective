import multiobjective.experiment as experiment
from multiobjective.config import Config, NSGAConfig, PSOConfig, GWOConfig
from multiobjective.algorithms.mogwo import run_mogwo
import multiobjective.algorithms as algorithms


def test_mogwo_handles_small_archive(monkeypatch):
    cfg = Config(
        num_times=1,
        num_services=4,
        coverage_fraction=0.1,
        nsga=NSGAConfig(population_size=4, max_generations=1),
        pso=PSOConfig(swarm_size=4, max_iterations=1),
        gwo=GWOConfig(wolf_size=2, max_iters=1),
    )

    monkeypatch.setattr(algorithms, "ALG_REGISTRY", {"mogwo": run_mogwo})

    class DummyStreak:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

    monkeypatch.setattr(experiment, "StreakTracker", DummyStreak)

    result = experiment.run_experiment(cfg)

    assert "mogwo" in result["series"]
