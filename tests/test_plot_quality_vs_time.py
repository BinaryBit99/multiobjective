import matplotlib
matplotlib.use("Agg")

from multiobjective.plotting import plot_quality_vs_time


def test_plot_quality_vs_time_runs():
    times = [0, 1]
    hv_series = [[[0.1, 0.2], [0.15, 0.25]]]
    time_series = [[[0.01, 0.03], [0.02, 0.04]]]
    plot_quality_vs_time(times, hv_series, time_series, ["alg"])  # Should not raise
