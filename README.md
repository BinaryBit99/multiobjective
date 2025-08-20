# Multiobjective Simulation

This repository provides tools for running multi-objective service selection
experiments. It includes several algorithms and helpers for analysing the
resulting error and cost trade-offs.

## Example visualisation

An example script demonstrates how to run a small experiment and plot the
results using `matplotlib`.

### Requirements

```bash
pip install matplotlib
```

### Running the example

From the repository root execute:

```bash
python examples/plot_results.py
```

The script runs `run_experiment` with a tiny configuration and displays error
and cost trends for two algorithms alongside their final error–cost tradeoff.
