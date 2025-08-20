# MULTIOBJ_ALGORITHMS/defaults.py
from .config import cfg
from .metrics.scs import OUParams   # OUParams is defined in metrics/scs.py

OU_PARAMS_DEFAULT = OUParams(
    theta=cfg.ou_theta,
    sigma=cfg.ou_sigma,
    delta_t=cfg.delta_t,
)


