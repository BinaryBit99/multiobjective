from typing import Dict, Callable
from .greedy import greedy_run
from .nsga2 import run_nsga2
from .mopso import run_mopso
from .mogwo import run_mogwo

ALG_REGISTRY: Dict[str, Callable] = {
    "greedy": greedy_run,
    "nsga":   run_nsga2,
    "mopso":  run_mopso,
    "mogwo":  run_mogwo,
}
