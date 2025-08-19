# config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import math

# --- helpers ---------------------------------------------------
_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
}
def _ratio_to_nums(ratio: str) -> tuple[int, int]:
    left, right = ratio.split("_")
    return _WORDS[left], _WORDS[right]

# --- algorithm-specific configs --------------------------------
@dataclass
class NSGAConfig:
    population_size: int = 120
    crossover_prob_min: float = 0.85
    crossover_prob_max: float = 0.95
    mutation_prob_min: float = 0.04
    mutation_prob_max: float = 0.08
    crossover_eta: float = 15.0
    mutation_eta: float  = 30.0
    tournament_size: int = 2
    max_generations: int = 300
    patience: int = 75

@dataclass
class PSOConfig:
    swarm_size: int = 100
    max_iterations: int = 120
    archive_size: int = 120
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 1.8
    c2: float = 1.8
    v_max: float = 4.0

@dataclass
class GWOConfig:
    wolf_size: int = 100
    max_iters: int = 150
    archive_size: int = 120

# --- top-level config ------------------------------------------
@dataclass
class Config:
    # simulation
    num_times: int = 30
    spatial_distribution: str = "uniform"   # 'uniform' | 'random' | 'clumped'
    space_size: Tuple[int,int] = (100, 100)
    num_clusters: int = 3
    cluster_spread: float = 10.0
    coverage_fraction: float = 0.20         # fraction of diagonal to use

    # services & ratio
    num_services: int = 100
    ratio_str: str = "three_one"            # providers_consumers ratio

    # QoS weights
    gamma_qos: float = 0.5
    lambda_vol: float = 0.25

    # OU process (for motion + look-ahead)
    ou_theta: float = 0.10
    ou_sigma: float = 5.0
    delta_t: float = 1.0

    # SCS(t+1) Monte Carlo look-ahead
    scs_lookahead_weight: float = 0.40      # set 0.0 to disable
    scs_mc_rollouts: int = 96               # 64–128 is typical

    # RNG
    master_seed: int = 42

    # algorithm configs
    nsga: NSGAConfig = field(default_factory=NSGAConfig)
    pso:  PSOConfig  = field(default_factory=PSOConfig)
    gwo:  GWOConfig  = field(default_factory=GWOConfig)

    # --- convenient computed properties ---
    @property
    def coverage_radius(self) -> float:
        w, h = self.space_size
        return self.coverage_fraction * math.sqrt(w*w + h*h)

    @property
    def num_providers(self) -> int:
        l, r = _ratio_to_nums(self.ratio_str)
        return round(self.num_services * (l / (l + r)))

    @property
    def num_consumers(self) -> int:
        return self.num_services - self.num_providers

# --- functional helpers -----------------------------------------
def coverage_radius(cfg: Config) -> float:
    """Return the coverage radius derived from the configuration."""
    w, h = cfg.space_size
    return cfg.coverage_fraction * math.sqrt(w * w + h * h)


def providers_consumers_from_ratio(cfg: Config) -> tuple[int, int]:
    """Compute the provider/consumer counts from ``cfg.ratio_str``."""
    l, r = _ratio_to_nums(cfg.ratio_str)
    num_providers = round(cfg.num_services * (l / (l + r)))
    num_consumers = cfg.num_services - num_providers
    return num_providers, num_consumers

# one shared instance you import everywhere
cfg = Config()

# --- (optional) legacy shims so old code still runs -------------
# If you still have modules referencing ALL_CAPS constants,
# these make them read from cfg without a big refactor.
NUM_TIMES            = cfg.num_times
SPATIAL_DISTRIBUTION = cfg.spatial_distribution
SPACE_SIZE           = cfg.space_size
NUM_CLUSTERS         = cfg.num_clusters
CLUSTER_SPREAD       = cfg.cluster_spread
COVERAGE_RADIUS      = cfg.coverage_radius
GAMMA_QOS            = cfg.gamma_qos
LAMBDA_VOL           = cfg.lambda_vol
