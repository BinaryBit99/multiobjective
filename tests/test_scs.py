import os
import sys
import numpy as np
from types import SimpleNamespace

# Ensure the repository root is on the import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from metrics.scs import (
    OUParams,
    SCSConfig,
    mc_coverage_prob,
    qos_success_prob,
    expected_pair_scs_tplus1,
    scs,
    expected_scs_next,
)


def coverage_radius(cfg) -> float:
    w, h = cfg.space_size
    return cfg.coverage_fraction * ((w * w + h * h) ** 0.5)


def make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        space_size=(10.0, 10.0),
        coverage_fraction=1.0,
        ou_theta=0.0,
        ou_sigma=0.0,
        delta_t=1.0,
        scs=SCSConfig(),
    )


def test_mc_coverage_prob_extremes():
    rng = np.random.default_rng(0)
    ou = OUParams(theta=0.0, sigma=0.0, delta_t=1.0)
    space = (10.0, 10.0)
    prob_same = mc_coverage_prob((1.0, 1.0), (1.0, 1.0), space, 1.0, ou, rng, K=32)

    rng = np.random.default_rng(0)
    prob_far = mc_coverage_prob((0.0, 0.0), (9.0, 9.0), space, 1.0, ou, rng, K=32)

    assert np.isclose(prob_same, 1.0)
    assert prob_far < 1e-6


def test_expected_pair_scs_tplus1():
    rng = np.random.default_rng(0)
    ou = OUParams(0.0, 0.0, 1.0)
    provider = {"coords": (0.0, 0.0), "qos": "Low"}
    consumer = {"coords": (0.0, 0.0)}
    T = {"Low": {"High": 0.2, "Medium": 0.3, "Low": 0.5}}

    p_qos = qos_success_prob("Low", transition_matrix=T)
    p_cov = mc_coverage_prob((0.0, 0.0), (0.0, 0.0), (10.0, 10.0), 1.0, ou, np.random.default_rng(0), K=32)

    expected = expected_pair_scs_tplus1(
        provider,
        consumer,
        (10.0, 10.0),
        1.0,
        ou,
        rng,
        transition_matrix=T,
        mc_rollouts=32,
    )

    assert np.isclose(expected, p_cov * p_qos)


def test_scs_basic_cases():
    cfg = make_cfg()
    scs_cfg = cfg.scs
    rng = np.random.default_rng(0)

    prods = [
        {"service_id": "p1", "coords": (0.0, 0.0), "qos": "High"},
        {"service_id": "p2", "coords": (0.0, 0.0), "qos": "High"},
        {"service_id": "p3", "coords": (0.0, 0.0), "qos": "Low"},
    ]
    cons = [{"service_id": "c1", "coords": (0.0, 0.0)}]
    prev_assign = {"c1": "p1"}

    score_same, _ = scs([0], (prods, cons), prev_assign, cfg, scs_cfg, rng)
    assert score_same == 1.0

    score_switched, _ = scs([1], (prods, cons), prev_assign, cfg, scs_cfg, rng)
    assert score_switched == 0.0

    score_bad, _ = scs([2], (prods, cons), prev_assign, cfg, scs_cfg, rng)
    assert score_bad == 0.0


def test_expected_scs_next_matches_pair():
    cfg = make_cfg()
    scs_cfg = cfg.scs
    rng = np.random.default_rng(0)
    rng_ref = np.random.default_rng(0)

    prods = [{"service_id": "p1", "coords": (0.0, 0.0), "qos": "High", "qos_prob": 1.0}]
    cons = [{"service_id": "c1", "coords": (0.0, 0.0)}]
    assign = [0]
    prev_assign = {"c1": "p1"}
    T = {"High": {"High": 1.0}}

    mean, _ = expected_scs_next(assign, (prods, cons), prev_assign, cfg, scs_cfg, rng, transition_matrix=T)

    ref = expected_pair_scs_tplus1(
        prods[0],
        cons[0],
        cfg.space_size,
        coverage_radius(cfg),
        OUParams(cfg.ou_theta, cfg.ou_sigma, cfg.delta_t),
        rng_ref,
        transition_matrix=T,
        mc_rollouts=scs_cfg.mc_samples,
    )

    assert 0.0 <= mean <= 1.0
    assert np.isclose(mean, ref)
