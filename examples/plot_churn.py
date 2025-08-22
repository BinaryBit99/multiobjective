"""Example script to visualise churn trends for different SCS weights."""
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

import numpy as np
from multiobjective.config import Config
from multiobjective.experiment import run_experiment
from multiobjective.plotting import plot_churn_over_time
from multiobjective.metrics import compute_churn
from multiobjective import algorithms
from multiobjective.algorithms.greedy import greedy_run


def main() -> None:
    """Run minimal experiments and plot churn for two SCS weights."""
    algorithms.ALG_REGISTRY = {"greedy": greedy_run}

    cfg = Config(num_times=5, num_services=8)
    results_w0 = run_experiment(cfg)
    results_w04 = run_experiment(replace(cfg, scs_lookahead_weight=0.4))

    times = list(range(1, cfg.num_times))
    alg = "greedy"
    churn_w0 = compute_churn(results_w0["series"][alg]["assignments"]["tp"])
    churn_w04 = compute_churn(results_w04["series"][alg]["assignments"]["tp"])
    scs_w0 = results_w0["series"][alg]["scs"]["tp"]["actual"][1:]
    scs_w04 = results_w04["series"][alg]["scs"]["tp"]["actual"][1:]

    plot_churn_over_time(
        times,
        [churn_w0, churn_w04],
        [scs_w0, scs_w04],
        ["w=0", "w=0.4"],
        f"{alg} churn over time",
    )


if __name__ == "__main__":
    main()
