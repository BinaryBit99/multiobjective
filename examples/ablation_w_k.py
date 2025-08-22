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

from pathlib import Path
import sys
from dataclasses import replace

# Ensure repository root is on path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from multiobjective.config import Config, NSGAConfig, PSOConfig
from multiobjective.experiment import run_experiment
from multiobjective.plotting import plot_sensitivity_grid
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

    plot_sensitivity_grid(results, weights, budgets, "NSGA-II")


if __name__ == "__main__":
    main()
