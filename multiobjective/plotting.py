import matplotlib.pyplot as plt
import numpy as np

def plot_metric_over_time(times, series, labels, title, ylabel):
    plt.figure(figsize=(10,4))
    for ys, lab in zip(series, labels):
        plt.plot(times, ys, marker="o", label=lab)
    plt.title(title); plt.xlabel("t"); plt.ylabel(ylabel); plt.grid(True); plt.legend(); plt.show()

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
