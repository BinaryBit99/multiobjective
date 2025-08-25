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
and cost trends for two algorithms alongside their error–cost tradeoff across
time.

### Plotting CLI results

To visualise the output from a CLI run saved to `results.json`, first execute:

```bash
python -m multiobjective.cli run --out results.json
```

Then load the JSON and generate plots:

```bash
python - <<'PY'
import json
from multiobjective.plotting import plot_metric_over_time, plot_tradeoff, plot_indicator_metric

# Load CLI output
with open("results.json") as f:
    results = json.load(f)

series = results["series"]
times = range(len(next(iter(series.values()))["errors"]["tp"]))
algs = list(series)

# Plot trajectories
plot_metric_over_time(times,
    [series[a]["errors"]["tp"] for a in algs],
    algs, "Topology error over time", "Normalised error")
plot_metric_over_time(times,
    [series[a]["costs"]["tp"] for a in algs],
    algs, "Cost over time", "Normalised cost")

# Plot a quality indicator (e.g. hypervolume)
plot_indicator_metric(
    results["indicators"],
    "HV",
    "tp",
    "Topology hypervolume over time",
    "Hypervolume",
)

# Plot trade-off over time
plot_tradeoff(
    [series[a]["errors"]["tp"] for a in algs],
    [series[a]["costs"]["tp"] for a in algs],
    algs, "Error–cost tradeoff over time")
PY
```

### Plotting indicator metrics

The experiment also logs common quality indicators such as hypervolume (HV),
inverted generational distance (IGD) and additive epsilon (EPS) for each
algorithm over time. The helper function `indicator_series` from
`multiobjective.plotting` extracts these values in a convenient form for
visualisation with `plot_metric_over_time`.

```bash
python - <<'PY'
import json
from multiobjective.plotting import indicator_series, plot_metric_over_time

with open("results.json") as f:
    results = json.load(f)

# Collect HV values for topology error across all algorithms
inds = results["indicators"]
times, hv_series, algs = indicator_series(inds, "HV", err_type="tp")

plot_metric_over_time(times, hv_series, algs,
                      "Hypervolume over time", "HV")
PY
```

The resulting plot shows how the hypervolume indicator evolves for each
algorithm, enabling direct comparison of their performance.
