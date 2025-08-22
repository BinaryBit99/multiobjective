import matplotlib.pyplot as plt
import numpy as np


def indicator_series(indicators: dict, name: str, err_type: str = "tp"):
    """Extract per-algorithm indicator values over time.

    Parameters
    ----------
    indicators : mapping
        The ``"indicators"`` portion of :func:`multiobjective.experiment.run_experiment`
        output. It maps algorithm names to indicator time series.
    name : str
        Indicator name such as ``"HV"`` (hypervolume), ``"IGD"`` or ``"EPS"``.
    err_type : str, optional
        Error category to select (``"tp"`` for topology or ``"res"`` for
        resilience errors).

    Returns
    -------
    times : range
        Time steps corresponding to the indicator values.
    series : list[list[float]]
        A list of indicator series, one per algorithm.
    labels : list[str]
        Algorithm labels matching ``series`` order.
    """

    labels = list(indicators)
    series = [indicators[a][err_type][name] for a in labels]
    times = range(len(series[0]) if series else 0)
    return times, series, labels


def plot_metric_over_time(times, series, labels, title, ylabel, caption: str | None = None):
    plt.figure(figsize=(10,4))
    for ys, lab in zip(series, labels):
        plt.plot(times, ys, marker="o", label=lab)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    if caption:
        plt.figtext(0.5, -0.05, caption, ha="center", va="center", fontsize=9)
    plt.show()

def plot_metric_with_std(times, series, stds, labels, title, ylabel):
    """Plot metric trajectories with mean and standard deviation.

    Parameters
    ----------
    times : sequence
        The x-axis values, typically representing time steps.
    series : sequence of sequences
        Mean metric values for each algorithm.
    stds : sequence of sequences
        Standard deviations corresponding to ``series``.
    labels : sequence of str
        Labels for each algorithm.
    title : str
        Title for the plot.
    ylabel : str
        Label for the y-axis.
    """
    plt.figure(figsize=(10,4))
    for ys, ss, lab in zip(series, stds, labels):
        plt.errorbar(times, ys, yerr=ss, marker="o", label=lab)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_tradeoff(errors, costs, labels, title):
    plt.figure(figsize=(8,6))
    mk = ["o","s","^","x","d","*"]
    for i, (e,c,l) in enumerate(zip(errors, costs, labels)):
        plt.scatter(e, c, marker=mk[i%len(mk)], label=l)
    plt.xlabel("Error"); plt.ylabel("Cost"); plt.title(title); plt.grid(True); plt.legend(); plt.show()
