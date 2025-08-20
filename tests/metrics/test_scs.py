import numpy as np
import pytest

from multiobjective.metrics.scs import (
    _ou_step,
    mc_coverage_prob,
    qos_success_prob,
    expected_pair_scs_tplus1,
    OUParams,
    SCSConfig,
)


def test_ou_step_moves_toward_mean(rng):
    x = np.array([0.0, 0.0])
    mu = np.array([1.0, 1.0])
    ou = OUParams(theta=1.0, sigma=0.0, delta_t=1.0)
    result = _ou_step(x, mu, ou, rng)
    assert np.allclose(result, mu)


def test_mc_coverage_prob_increases_with_radius():
    kwargs = dict(
        p_coords=(0.0, 0.0),
        c_coords=(5.0, 0.0),
        space_size=(10.0, 10.0),
        ou=OUParams(theta=0.0, sigma=1.0, delta_t=1.0),
        K=256,
    )
    rng_small = np.random.default_rng(42)
    rng_large = np.random.default_rng(42)
    p_small = mc_coverage_prob(radius=1.0, rng=rng_small, **kwargs)
    p_large = mc_coverage_prob(radius=3.0, rng=rng_large, **kwargs)
    assert p_small <= p_large


def test_qos_success_prob():
    tm = {"Low": {"Low": 0.1, "Medium": 0.6, "High": 0.3}}
    prob = qos_success_prob("Low", tm)
    assert prob == pytest.approx(0.9)


def test_expected_pair_scs_matches_product(cfg):
    p = {
        "coords": (0.0, 0.0),
        "qos": "Low",
        "qos_prob": 0.2,
        "response_time_ms": 1,
        "throughput_kbps": 1,
    }
    c = {
        "coords": (5.0, 0.0),
        "response_time_ms": 1,
        "throughput_kbps": 1,
    }
    tm = {"Low": {"Medium": 0.6, "High": 0.3}}
    ou = OUParams(theta=0.0, sigma=1.0, delta_t=1.0)
    radius = 1.0
    rng_cov = np.random.default_rng(7)
    rng_pair = np.random.default_rng(7)
    p_cov = mc_coverage_prob(
        p_coords=p["coords"],
        c_coords=c["coords"],
        space_size=cfg.space_size,
        radius=radius,
        ou=ou,
        rng=rng_cov,
        K=128,
    )
    p_qos = qos_success_prob("Low", tm, fallback_prob=0.2)
    combined = expected_pair_scs_tplus1(
        provider_record=p,
        consumer_record=c,
        space_size=cfg.space_size,
        radius=radius,
        ou=ou,
        rng=rng_pair,
        transition_matrix=tm,
        mc_rollouts=128,
    )
    assert combined == pytest.approx(p_qos * p_cov)


def test_scs_config_weight_bounds():
    with pytest.raises(ValueError):
        SCSConfig(weight=-0.1)
    with pytest.raises(ValueError):
        SCSConfig(weight=1.1)


def test_scs_config_mc_samples_positive():
    with pytest.raises(ValueError):
        SCSConfig(mc_samples=0)
