"""Example script to run a small experiment and plot results.

This script runs :func:`multiobjective.experiment.run_experiment` with a
lightweight configuration and produces a few basic visualisations of the
resulting metrics using the helper functions in :mod:`multiobjective.plotting`.
"""
import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend for example usage

from pathlib import Path
import sys

# Ensure the project root's parent is on the Python path and avoid name clashes
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
_parent = _repo_root.parent
for p in ("", str(_script_dir), str(_repo_root)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.append(str(_parent))

import numpy as np
from multiobjective.config import Config, NSGAConfig, PSOConfig, GWOConfig, coverage_radius
from multiobjective.experiment import run_experiment
from multiobjective.plotting import (
    plot_metric_over_time,
    plot_metric_with_std,
    plot_tradeoff,
)
from multiobjective.simulation import euclidean_distance
from multiobjective.metrics.scs import blended_error
from multiobjective.defaults import OU_PARAMS_DEFAULT
from multiobjective import algorithms


def main() -> None:
    """Execute a minimal experiment and display several plots."""
    # Limit the algorithm registry so the example stays quick to run.
    # Disable heavy SCS computations for the example.
    def _noop_scs(*args, **kwargs):
        return 0.0, None

    _experiment_module = __import__("multiobjective.experiment", fromlist=["scs"])
    _experiment_module.scs = _noop_scs
    _experiment_module.expected_scs_next = lambda *a, **k: (0.0, None)

    # Define two lightweight algorithms for demonstration purposes.
    def random_match(cfg, rng_pool, records, cost_per, err_type, metrics, _streaks, norm_fn):
        errors, costs, stds = [], [], []
        radius = coverage_radius(cfg)
        for t in range(cfg.num_times):
            rng = rng_pool.for_("greedy", t)
            scs_rng = rng_pool.for_("scs", t)
            prods, cons = records[t]
            curr_max, curr_min = max(cost_per[f"{t}"]), min(cost_per[f"{t}"])
            matched = []
            for c in cons:
                viable = [p for p in prods if euclidean_distance(p, c) <= radius]
                if not viable:
                    viable = prods  # fall back to any provider
                p = rng.choice(viable)
                err = blended_error(err_type, p, c, t, cfg, norm_fn, scs_rng,
                                    ou_params=OU_PARAMS_DEFAULT)
                cost_norm = (p.cost - curr_min) / (curr_max - curr_min + 1e-12)
                matched.append((err, cost_norm))
            errs = [m[0] for m in matched]; cs = [m[1] for m in matched]
            errors.append(float(np.mean(errs)))
            costs.append(float(np.mean(cs)))
            stds.append(float(np.std(errs)) if len(errs) > 1 else 0.0)
            metrics.record("random", err_type, t, [(errors[-1], costs[-1])])
        return errors, costs, stds

    def min_cost_match(cfg, rng_pool, records, cost_per, err_type, metrics, _streaks, norm_fn):
        errors, costs, stds = [], [], []
        radius = coverage_radius(cfg)
        for t in range(cfg.num_times):
            scs_rng = rng_pool.for_("scs", t)
            prods, cons = records[t]
            curr_max, curr_min = max(cost_per[f"{t}"]), min(cost_per[f"{t}"])
            matched = []
            for c in cons:
                viable = [p for p in prods if euclidean_distance(p, c) <= radius]
                if not viable:
                    viable = prods
                p = min(viable, key=lambda x: x.cost)
                err = blended_error(err_type, p, c, t, cfg, norm_fn, scs_rng,
                                    ou_params=OU_PARAMS_DEFAULT)
                cost_norm = (p.cost - curr_min) / (curr_max - curr_min + 1e-12)
                matched.append((err, cost_norm))
            errs = [m[0] for m in matched]; cs = [m[1] for m in matched]
            errors.append(float(np.mean(errs)))
            costs.append(float(np.mean(cs)))
            stds.append(float(np.std(errs)) if len(errs) > 1 else 0.0)
            metrics.record("min_cost", err_type, t, [(errors[-1], costs[-1])])
        return errors, costs, stds

    algorithms.ALG_REGISTRY = {"random": random_match, "min_cost": min_cost_match}

    cfg = Config(
        num_times=5,
        num_services=10,
        ratio_str="one_one",
        scs_lookahead_weight=0.0,
        scs_mc_rollouts=4,
        nsga=NSGAConfig(population_size=10, max_generations=5, patience=5),
        pso=PSOConfig(swarm_size=10, max_iterations=5, archive_size=10),
        gwo=GWOConfig(wolf_size=10, max_iters=5, archive_size=10),
    )

    results = run_experiment(cfg)
    series = results["series"]
    times = list(range(cfg.num_times))

    # Choose the algorithms to visualise
    algs = ["random", "min_cost"]
    error_series = [series[a]["errors"]["tp"] for a in algs]
    std_series = [series[a]["stds"]["tp"] for a in algs]
    cost_series = [series[a]["costs"]["tp"] for a in algs]
    res_error_series = [series[a]["errors"]["res"] for a in algs]
    res_cost_series = [series[a]["costs"]["res"] for a in algs]

    # Plot error and cost trajectories (error with variability)
    plot_metric_with_std(
        times,
        error_series,
        std_series,
        algs,
        "Topology error over time",
        "Normalised error",
    )
    plot_metric_over_time(times, cost_series, algs, "Cost over time", "Normalised cost")
    plot_metric_over_time(times, res_error_series, algs, "Resource error over time", "Normalised error")
    plot_metric_over_time(times, res_cost_series, algs, "Resource cost over time", "Normalised cost")

    # Plot actual vs expected SCS for topology and resilience errors
    scs_tp = results["scs"]["tp"]
    scs_E_tp = results["scs"]["E_tp"]
    scs_res = results["scs"]["res"]
    scs_E_res = results["scs"]["E_res"]

    plot_metric_over_time(
        times,
        [scs_tp, scs_E_tp],
        ["Actual SCS", "Expected SCS"],
        "Topology continuity over time",
        "Continuity score",
        caption="Overlay of actual vs expected service continuity for topology errors",
    )

    plot_metric_over_time(
        times,
        [scs_res, scs_E_res],
        ["Actual SCS", "Expected SCS"],
        "Resilience continuity over time",
        "Continuity score",
        caption="Overlay of actual vs expected service continuity for resilience errors",
    )

    # Show error–cost tradeoffs for the final time step
    final_errors = [errs[-1] for errs in error_series]
    final_costs = [cs[-1] for cs in cost_series]
    plot_tradeoff(final_errors, final_costs, algs, "Error–Cost tradeoff at final step")

    res_final_errors = [errs[-1] for errs in res_error_series]
    res_final_costs = [cs[-1] for cs in res_cost_series]
    plot_tradeoff(res_final_errors, res_final_costs, algs, "Resource error–cost tradeoff at final step")


if __name__ == "__main__":
    main()
