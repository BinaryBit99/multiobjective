"""Ablation study over SCS weight and Monte Carlo budget.

This script runs :func:`multiobjective.experiment.run_experiment` for several
values of the SCS look-ahead weight ``w`` and Monte Carlo budget ``K``.  The
resulting metrics are visualised in a 2×2 grid:

* **Panel A** – final Pareto fronts for different ``w`` values.
* **Panel B** – service-continuity score (SCS) at matched error quantiles
  versus ``w``.
* **Panel C** – ranking stability across ``K`` budgets measured by Kendall's
  ``τ``.
* **Panel D** – wall-clock runtime averaged over ``w`` for each ``K``.

The configuration is intentionally small so the example runs quickly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
from scipy.stats import kendalltau

# Ensure repository root is on path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from multiobjective.config import Config, NSGAConfig, PSOConfig
from multiobjective.experiment import run_experiment
from multiobjective.plotting import (
    plot_pareto_front_shift,
    plot_scs_vs_weight_at_error,
    create_2x2_figure,
)
from multiobjective import algorithms
from multiobjective.algorithms.nsga2 import run_nsga2
from multiobjective.algorithms.mopso import run_mopso


def main() -> None:
    weights = [0.0, 0.2, 0.4, 0.6]
    budgets = [32, 64, 128, 256]

    # Limit algorithms to keep runtime short
    algorithms.ALG_REGISTRY = {"nsga": run_nsga2, "mopso": run_mopso}

    base_cfg = Config(
        num_times=5,
        num_services=8,
        scs_mc_rollouts=budgets[0],
        nsga=NSGAConfig(population_size=20, max_generations=10, patience=5),
        pso=PSOConfig(swarm_size=20, max_iterations=10, archive_size=20),
    )

    results: dict[float, dict[int, dict]] = {}
    for w in weights:
        results[w] = {}
        for K in budgets:
            cfg = replace(base_cfg, scs_lookahead_weight=w, scs_mc_rollouts=K)
            results[w][K] = run_experiment(cfg)

    # Create figure grid
    fig, axes = create_2x2_figure(figsize=(12, 10))
    (axA, axB), (axC, axD) = axes

    # Panel A – Pareto fronts for different w
    plt.sca(axA)
    alg = "nsga"
    last_t = str(base_cfg.num_times - 1)
    front_w0 = results[0.0][budgets[-1]]["fronts"][alg]["tp"][last_t]
    for w in weights[1:]:
        front_w = results[w][budgets[-1]]["fronts"][alg]["tp"][last_t]
        plot_pareto_front_shift(front_w0, front_w, "NSGA-II")
    axA.set_title("Front shift vs w")

    # Panel B – SCS at matched error quantiles vs w
    plt.sca(axB)
    q_vals = [0.25, 0.5, 0.75]
    scs_w = {w: [] for w in weights}
    for w in weights:
        series = results[w][budgets[-1]]["series"][alg]
        errs = np.array(series["errors"]["tp"])
        scs_vals = np.array(series["scs"]["tp"]["actual"])
        for q in q_vals:
            q_err = np.quantile(errs, q)
            idx = int(np.argmin(np.abs(errs - q_err)))
            scs_w[w].append(scs_vals[idx])
    plot_scs_vs_weight_at_error(
        scs_w,
        weights,
        [f"q={q}" for q in q_vals],
        "SCS vs w at error quantiles",
    )

    # Panel C – ranking stability vs K
    plt.sca(axC)
    hv_per_K = {}
    w0 = weights[0]
    for K in budgets:
        ind = results[w0][K]["indicators"]
        hv_per_K[K] = {a: ind[a]["tp"]["HV"][-1] for a in ind}
    base_rank = sorted(hv_per_K[budgets[-1]], key=hv_per_K[budgets[-1]].get, reverse=True)
    base_ranks = {alg: i for i, alg in enumerate(base_rank)}
    taus = []
    for K in budgets:
        rank = sorted(hv_per_K[K], key=hv_per_K[K].get, reverse=True)
        ranks = {alg: i for i, alg in enumerate(rank)}
        a1 = [base_ranks[a] for a in base_rank]
        a2 = [ranks[a] for a in base_rank]
        tau, _ = kendalltau(a1, a2)
        taus.append(tau)
    plt.plot(budgets, taus, marker="o")
    plt.xlabel("K")
    plt.ylabel("Kendall tau")
    plt.title("Ranking stability vs K")
    plt.grid(True)

    # Panel D – runtime vs K (average over weights)
    plt.sca(axD)
    runtimes = []
    for K in budgets:
        rt = [results[w][K]["meta"]["runtime_s"] for w in weights]
        runtimes.append(sum(rt) / len(rt))
    plt.plot(budgets, runtimes, marker="o")
    plt.xlabel("K")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs K")
    plt.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
