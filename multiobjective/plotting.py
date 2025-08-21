import matplotlib.pyplot as plt
import numpy as np


def plot_metric_over_time(times, series, labels, title, ylabel, ax=None):
    """Plot a generic metric's trajectory for multiple algorithms.

    Parameters
    ----------
    times:
        Sequence of time points.
    series:
        Sequence of ``y`` value sequences – one per algorithm.
    labels:
        Algorithm labels corresponding to ``series``.
    title:
        Title for the plot.
    ylabel:
        Label for the y axis.
    ax:
        Optional matplotlib axis to plot on. When ``None`` (default) a new
        figure and axis are created and ``plt.show`` is called.
    """

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    for ys, lab in zip(series, labels):
        ax.plot(times, ys, marker="o", label=lab)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    if created_fig:
        plt.show()
    return ax

def plot_tradeoff(errors, costs, labels, title):
    plt.figure(figsize=(8, 6))
    mk = ["o", "s", "^", "x", "d", "*"]
    for i, (e, c, l) in enumerate(zip(errors, costs, labels)):
        plt.scatter(e, c, marker=mk[i % len(mk)], label=l)
    plt.xlabel("Error")
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_indicators_over_time(times, indicators, algs, aggregate=False):
    """Visualise HV/IGD/EPS indicators for each algorithm over time.

    Parameters
    ----------
    times:
        Sequence of time points.
    indicators:
        Mapping of algorithm -> error-type -> indicator name -> values.
    algs:
        Iterable of algorithm names to include in the plots.
    aggregate:
        When ``True``, plot all indicators in a single multi-panel figure.
        Otherwise, individual figures are produced via
        :func:`plot_metric_over_time`.
    """

    metrics = ["HV", "IGD", "EPS"]
    err_types = ["tp", "res"]

    if aggregate:
        fig, axes = plt.subplots(
            len(err_types),
            len(metrics),
            figsize=(5 * len(metrics), 3 * len(err_types)),
            sharex=True,
        )
        for r, te in enumerate(err_types):
            for c, m in enumerate(metrics):
                ax = axes[r, c]
                series = [indicators[a][te][m] for a in algs]
                plot_metric_over_time(times, series, algs, f"{m} ({te})", m, ax=ax)
        fig.tight_layout()
        plt.show()
    else:
        for te in err_types:
            for m in metrics:
                series = [indicators[a][te][m] for a in algs]
                plot_metric_over_time(
                    times,
                    series,
                    algs,
                    f"{m} ({te}) over time",
                    m,
                )

