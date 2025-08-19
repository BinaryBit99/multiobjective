import numpy as np, math
from .config import Config
from .rng import RNGPool

def OU_step(x_t, mu, theta, sigma, delta_t, rng):
    noise = rng.standard_normal(size=x_t.shape)
    return x_t + theta*(mu-x_t)*delta_t + sigma*np.sqrt(delta_t)*noise

def generate_OU_trajectory(start_pos, mu, num_steps, theta, sigma, delta_t, rng):
    traj = [start_pos]
    for _ in range(num_steps):
        nxt = OU_step(traj[-1], mu, theta, sigma, delta_t, rng)
        nxt = np.clip(nxt, [0.,0.], [float(_SPACE[0]), float(_SPACE[1])])
        traj.append(nxt)
    return np.array(traj)

def init_node_positions_and_means(total_nodes, distribution, space, num_clusters, cluster_spread, rng):
    starts, means = [], []
    if distribution == "uniform":
        for _ in range(total_nodes):
            pos = rng.uniform(low=[0.,0.], high=[float(space[0]), float(space[1])], size=(2,))
            starts.append(pos); means.append(np.array(space, dtype=float)/2.0)
    elif distribution == "random":
        for _ in range(total_nodes):
            pos = rng.random(2) * np.array(space, dtype=float)
            starts.append(pos); means.append(np.array(space, dtype=float)/2.0)
    elif distribution == "clumped":
        centers = rng.random((num_clusters,2))*np.array(space,dtype=float)
        for _ in range(total_nodes):
            cen = centers[rng.integers(0, num_clusters)]
            pos = rng.normal(loc=cen, scale=cluster_spread, size=2)
            pos = np.clip(pos, [0.,0.], [float(space[0]), float(space[1])])
            starts.append(pos); means.append(cen)
    else:
        raise ValueError(f"Unknown distribution {distribution}")
    return np.array(starts), np.array(means)

def euclidean_distance(p: dict, c: dict) -> float:
    (px,py), (cx,cy) = p["coords"], c["coords"]
    return math.sqrt((px-cx)**2 + (py-cy)**2)

_SPACE = (100,100)  # overwritten by build_trajectories

def build_trajectories(cfg: Config, rng_pool: RNGPool, num_providers: int, num_consumers: int) -> dict[str, np.ndarray]:
    global _SPACE
    _SPACE = cfg.space_size
    total = num_providers + num_consumers
    starts, mus = init_node_positions_and_means(
        total, cfg.spatial_distribution, cfg.space_size, cfg.num_clusters, cfg.cluster_spread,
        rng_pool.for_("ou_init")
    )
    out = {}
    for i in range(total):
        rng = rng_pool.for_("ou_node", idx=i)
        traj = generate_OU_trajectory(
            starts[i], mus[i], cfg.num_times, cfg.ou_theta, cfg.ou_sigma, cfg.delta_t, rng
        )
        sid = f"p{i}" if i < num_providers else f"c{i-num_providers}"
        out[sid] = traj
    return out
