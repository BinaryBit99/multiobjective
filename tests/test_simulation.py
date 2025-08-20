import numpy as np
from multiobjective.simulation import generate_OU_trajectory, build_trajectories
from multiobjective.rng import RNGPool


def test_generate_ou_trajectory_bounds():
    space = (5, 5)
    start = np.array([2.0, 2.0])
    mu = np.array(space, dtype=float) / 2.0
    rng = np.random.default_rng(0)
    traj = generate_OU_trajectory(
        start, mu, num_steps=10, theta=0.1, sigma=5.0, delta_t=1.0, rng=rng, space=space
    )
    assert np.all(traj[:, 0] >= 0) and np.all(traj[:, 0] <= space[0])
    assert np.all(traj[:, 1] >= 0) and np.all(traj[:, 1] <= space[1])


def test_build_trajectories_bounds(cfg):
    rng_pool = RNGPool(cfg.master_seed, cfg.num_times)
    trajs = build_trajectories(cfg, rng_pool, num_providers=2, num_consumers=3)
    w, h = cfg.space_size
    for traj in trajs.values():
        assert np.all(traj[:, 0] >= 0) and np.all(traj[:, 0] <= w)
        assert np.all(traj[:, 1] >= 0) and np.all(traj[:, 1] <= h)
