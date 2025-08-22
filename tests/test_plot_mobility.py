import matplotlib
matplotlib.use("Agg")

import numpy as np

from multiobjective.config import Config
from multiobjective.data import RecordBuilder
from multiobjective.rng import RNGPool
from multiobjective.plotting import (
    collect_displacements,
    lag1_autocorrelation,
    plot_mobility_checks,
)


def test_collect_and_plot_mobility():
    cfg = Config(num_times=5, num_services=4, ratio_str="two_two")
    cfg.__post_init__()
    rng_pool = RNGPool(cfg.master_seed, cfg.num_times)
    records, _, _, _, _ = RecordBuilder(cfg, rng_pool)

    info = collect_displacements(records)
    assert info["providers"].size == cfg.num_providers * (cfg.num_times - 1)
    assert info["consumers"].size == cfg.num_consumers * (cfg.num_times - 1)

    ac = lag1_autocorrelation(info["providers_pos"])
    assert np.isfinite(ac)

    fig, ax = plot_mobility_checks(info, cfg.ou_theta, "mobility")
    assert fig is not None and ax is not None

