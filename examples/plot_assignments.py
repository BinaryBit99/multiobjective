"""Example script to visualise assignment heatmaps for two SCS weights."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import sys
from dataclasses import replace

# Ensure repository root is on path for local execution
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from multiobjective.config import Config
from multiobjective.experiment import run_experiment
from multiobjective.plotting import plot_assignment_heatmap
from multiobjective import algorithms
from multiobjective.algorithms.greedy import greedy_run


def main() -> None:
    """Run minimal experiments and plot assignments for two SCS weights."""

    algorithms.ALG_REGISTRY = {"greedy": greedy_run}

    cfg = Config(num_times=5, num_services=8)
    results_w0 = run_experiment(cfg)
    results_w04 = run_experiment(replace(cfg, scs_lookahead_weight=0.4))

    alg = "greedy"
    assigns_w0 = results_w0["series"][alg]["assignments"]["tp"]
    assigns_w04 = results_w04["series"][alg]["assignments"]["tp"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axes[0])
    plot_assignment_heatmap(assigns_w0, "w=0")
    plt.sca(axes[1])
    plot_assignment_heatmap(assigns_w04, "w=0.4")
    fig.suptitle("Assignment heatmaps")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

