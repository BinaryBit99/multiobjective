import matplotlib
matplotlib.use("Agg")

from multiobjective.plotting import plot_assignment_heatmap


def test_plot_assignment_heatmap_runs():
    assignments = [[0, 1], [1, 0], [0, 1]]
    plot_assignment_heatmap(assignments, "assignments")

