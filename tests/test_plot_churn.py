import matplotlib
matplotlib.use("Agg")

from multiobjective.plotting import plot_churn_over_time


def test_plot_churn_over_time_runs():
    times = [1, 2, 3]
    churn = [[0.2, 0.3, 0.1]]
    scs = [[0.9, 0.85, 0.8]]
    plot_churn_over_time(times, churn, scs, ["alg"], "churn")
