"""
MULTIOBJ_ALGORITHMS package
Public surface.
"""
from .config import Config, NSGAConfig, PSOConfig, GWOConfig, coverage_radius, providers_consumers_from_ratio
from .metrics.scs import scs, expected_scs_next, SCSConfig, SCSComponents

# Optional imports that may pull heavy dependencies
try:  # pragma: no cover
    from .experiment import run_experiment
    from .algorithms import ALG_REGISTRY
except Exception:  # pragma: no cover
    run_experiment = None
    ALG_REGISTRY = {}

__all__ = [
    "Config", "NSGAConfig", "PSOConfig", "GWOConfig",
    "coverage_radius", "providers_consumers_from_ratio",
    "run_experiment",
    "SCSConfig", "SCSComponents", "scs", "expected_scs_next",
    "ALG_REGISTRY",
]

__version__ = "0.1.0"



