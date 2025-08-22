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


def plot_scs_over_time(times, series, labels, title, caption: str | None = None):
    """Plot service-continuity scores over time.

    This is a thin wrapper around :func:`plot_metric_over_time` with the
    y-axis label preset to ``"SCS"``.

    Parameters
    ----------
    times : sequence
        X-axis values (time steps).
    series : sequence of sequences
        SCS values for each labelled series.
    labels : sequence of str
        Labels corresponding to ``series``.
    title : str
        Title for the plot.
    caption : str, optional
        Caption placed below the plot.
    """

    plot_metric_over_time(times, series, labels, title, ylabel="SCS", caption=caption)

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


def indicator_metric_series(indicators: dict, metric: str, err_type: str):
    """Extract per-algorithm series for a given indicator metric.

    Parameters
    ----------
    indicators : dict
        The ``indicators`` section returned by :func:`run_experiment`.
    metric : str
        Indicator metric name (e.g. ``"HV"`` or ``"IGD"``).
    err_type : str
        Error type to extract (``"tp"`` or ``"res"``).

    Returns
    -------
    times : list[int]
        List of time indices.
    series : list[list[float]]
        Metric values for each algorithm.
    labels : list[str]
        Algorithm names corresponding to ``series`` order.
    """

    labels = sorted(indicators.keys())
    series = [indicators[a][err_type][metric] for a in labels]
    times = list(range(len(series[0]) if series else 0))
    return times, series, labels


def plot_indicator_metric(indicators: dict, metric: str, err_type: str,
                          title: str, ylabel: str, caption: str | None = None):
    """Plot an indicator metric over time for each algorithm.

    This is a convenience wrapper around :func:`plot_metric_over_time` that
    extracts the per-algorithm series for ``metric`` and ``err_type``.

    Parameters
    ----------
    indicators : dict
        The ``indicators`` section returned by :func:`run_experiment`.
    metric : str
        Indicator metric name (e.g. ``"HV"``).
    err_type : str
        Error type to extract (``"tp"`` or ``"res"``).
    title : str
        Title for the plot.
    ylabel : str
        Label for the y-axis.
    caption : str, optional
        Additional caption displayed below the plot.
    """

    times, series, labels = indicator_metric_series(indicators, metric, err_type)
    plot_metric_over_time(times, series, labels, title, ylabel, caption)


def plot_pareto_front_shift(front_w0, front_w1, alg_label):
    """Visualise the Pareto front change between two SCS weights.

    Parameters
    ----------
    front_w0 : sequence of tuple
        Final Pareto-optimal points when ``scs_lookahead_weight`` is ``0``.
    front_w1 : sequence of tuple
        Final Pareto-optimal points when ``scs_lookahead_weight`` is ``0.4``.
    alg_label : str
        Label describing the algorithm whose fronts are plotted.
    """

    from .pareto import nondominated_indices

    pts = [(e, c, "w=0") for e, c in front_w0]
    pts.extend((e, c, "w=0.4") for e, c in front_w1)
    objs = [(e, c) for e, c, _ in pts]
    nd = set(nondominated_indices(objs))

    seen = set()
    for i, (e, c, lab) in enumerate(pts):
        alpha = 1.0 if i in nd else 0.2
        color = "C0" if lab == "w=0" else "C1"
        marker = "o" if lab == "w=0" else "s"
        lbl = lab if lab not in seen else None
        seen.add(lab)
        plt.scatter(e, c, color=color, marker=marker, alpha=alpha, label=lbl)

    plt.xlabel("Error")
    plt.ylabel("Cost")
    plt.title(alg_label)
    plt.grid(True)
    plt.legend()
