import pytest

from multiobjective.algorithms.base import Individual
from multiobjective.rng import RNGPool


def norm_fn(kind, err, t):
    return err


def test_individual_evaluation_reproducible(cfg):
    prods = [
        {
            "coords": (0.0, 0.0),
            "response_time_ms": 1,
            "throughput_kbps": 1,
            "cost": 1.0,
            "qos_prob": 0.5,
            "qos_volatility": 0.0,
        }
    ]
    cons = [
        {
            "coords": (0.0, 0.0),
            "response_time_ms": 1,
            "throughput_kbps": 1,
        }
    ]

    def run(run_order):
        pool = RNGPool(master_seed=123, num_times=1)
        individuals = {"a": Individual([0]), "b": Individual([0])}
        for key in run_order:
            individuals[key].evaluate(
                prods,
                cons,
                "rel",
                norm_fn,
                0,
                cfg.gamma_qos,
                cfg.lambda_vol,
                cfg,
                pool,
            )
        return individuals["a"].error, individuals["b"].error

    first_run = run(["a", "b"])
    second_run = run(["b", "a"])
    assert first_run == pytest.approx(second_run)
