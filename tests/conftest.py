import numpy as np
import pytest

from multiobjective.config import Config
from multiobjective.metrics.scs import SCSConfig

@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.fixture
def scs_config():
    return SCSConfig(enabled=True, weight=0.5, mc_samples=4)

@pytest.fixture
def cfg(scs_config):
    cfg = Config(space_size=(10, 10), coverage_fraction=0.2)
    cfg.scs = scs_config
    return cfg
