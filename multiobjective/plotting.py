import matplotlib.pyplot as plt
import numpy as np


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

def plot_tradeoff(errors, costs, labels, title):
    plt.figure(figsize=(8,6))
    mk = ["o","s","^","x","d","*"]
    for i, (e,c,l) in enumerate(zip(errors, costs, labels)):
        plt.scatter(e, c, marker=mk[i%len(mk)], label=l)
    plt.xlabel("Error"); plt.ylabel("Cost"); plt.title(title); plt.grid(True); plt.legend(); plt.show()
