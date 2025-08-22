"""Plot Pareto-front changes for different SCS weights.

This example expects result files produced by
``multiobjective.experiment.run_experiment`` for two settings of
``Config.scs_lookahead_weight`` (``0`` and ``0.4``).  For each algorithm the
final Pareto fronts are plotted side-by-side to illustrate the effect of the
look-ahead term.
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for example scripts
import matplotlib.pyplot as plt

# Ensure repository root is on path for local execution
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from multiobjective.plotting import plot_pareto_front_shift


def _load_final_front(path: Path, alg: str):
    data = json.loads(path.read_text())
    fronts = data["fronts"][alg]["tp"]
    last_t = max(int(t) for t in fronts.keys())
    return fronts[str(last_t)]


def main() -> None:
    base = Path(__file__).resolve().parent
    algs = [("nsga", "NSGA-II"), ("mopso", "MOPSO")]
    fig, axes = plt.subplots(1, len(algs), figsize=(5 * len(algs), 4))
    for ax, (name, label) in zip(axes, algs):
        f0 = _load_final_front(base / f"{name}_w0.json", name)
        f1 = _load_final_front(base / f"{name}_w04.json", name)
        plt.sca(ax)
        plot_pareto_front_shift(f0, f1, label)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

