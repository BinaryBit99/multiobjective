import numpy as np
import pytest
import importlib
from multiobjective.types import ProviderRecord, ConsumerRecord
scs_module = importlib.import_module("multiobjective.metrics.scs")

from multiobjective.metrics.scs import (
    _ou_step,
    mc_coverage_prob,
    qos_success_prob,
    expected_pair_scs_tplus1,
    expected_scs_next,
    scs,
    OUParams,
)


def test_ou_step_moves_toward_mean(rng):
    x = np.array([0.0, 0.0])
    mu = np.array([1.0, 1.0])
    ou = OUParams(theta=1.0, sigma=0.0, delta_t=1.0)
    result = _ou_step(x, mu, ou, rng)
    assert np.allclose(result, mu)


def test_mc_coverage_prob_increases_with_radius():
    def mc_cov_loop(radius, rng, *, p_coords, c_coords, space_size, ou, K):
        p = np.asarray(p_coords, dtype=float)
        c = np.asarray(c_coords, dtype=float)
        center = np.array(space_size, dtype=float) / 2.0
        hits = 0
        for _ in range(K):
            p1 = np.clip(_ou_step(p, center, ou, rng), (0.0, 0.0), space_size)
            c1 = np.clip(_ou_step(c, center, ou, rng), (0.0, 0.0), space_size)
            if np.linalg.norm(p1 - c1) <= radius:
                hits += 1
        return hits / K

    kwargs = dict(
        p_coords=(0.0, 0.0),
        c_coords=(5.0, 0.0),
        space_size=(10.0, 10.0),
        ou=OUParams(theta=0.0, sigma=1.0, delta_t=1.0),
        K=256,
    )
    rng_small_vec = np.random.default_rng(42)
    rng_small_loop = np.random.default_rng(42)
    p_small = mc_coverage_prob(radius=1.0, rng=rng_small_vec, **kwargs)
    p_small_ref = mc_cov_loop(radius=1.0, rng=rng_small_loop, **kwargs)

    rng_large_vec = np.random.default_rng(42)
    rng_large_loop = np.random.default_rng(42)
    p_large = mc_coverage_prob(radius=3.0, rng=rng_large_vec, **kwargs)
    p_large_ref = mc_cov_loop(radius=3.0, rng=rng_large_loop, **kwargs)

    assert p_small <= p_large
    assert p_small == pytest.approx(p_small_ref)
    assert p_large == pytest.approx(p_large_ref)
    assert p_small == pytest.approx(0.0)
    assert p_large == pytest.approx(0.078125)


def test_qos_success_prob():
    tm = {"Low": {"Low": 0.1, "Medium": 0.6, "High": 0.3}}
    prob = qos_success_prob("Low", tm)
    assert prob == pytest.approx(0.9)


def test_expected_pair_scs_matches_product(cfg, monkeypatch):
    p = ProviderRecord(
        service_id="p0",
        timestamp=0,
        response_time_ms=1,
        throughput_kbps=1,
        cost=0.0,
        coords=(0.0, 0.0),
        qos="Low",
        qos_prob=0.2,
        qos_volatility=0.0,
    )
    c = ConsumerRecord(
        service_id="c0",
        timestamp=0,
        response_time_ms=1,
        throughput_kbps=1,
        cost=0.0,
        coords=(5.0, 0.0),
        qos=None,
        qos_prob=0.0,
        qos_volatility=0.0,
    )
    tm = {"Low": {"Medium": 0.6, "High": 0.3}}
    ou = OUParams(theta=0.0, sigma=1.0, delta_t=1.0)
    radius = 1.0
    call_count = {"n": 0}
    original = scs_module.mc_coverage_prob
    def wrapped_mc(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(scs_module, "mc_coverage_prob", wrapped_mc)

    rng_ref = np.random.default_rng(7)
    rng_pair = np.random.default_rng(7)
    p_cov_ref = original(
        p_coords=p.coords,
        c_coords=c.coords,
        space_size=cfg.space_size,
        radius=radius,
        ou=ou,
        rng=rng_ref,
        K=128,
    )
    cov_prob, qos_prob = expected_pair_scs_tplus1(
        provider_record=p,
        consumer_record=c,
        space_size=cfg.space_size,
        radius=radius,
        ou=ou,
        rng=rng_pair,
        transition_matrix=tm,
        mc_rollouts=128,
    )
    assert call_count["n"] == 1
    p_qos_ref = qos_success_prob("Low", tm, fallback_prob=0.2)
    assert cov_prob == pytest.approx(p_cov_ref)
    assert qos_prob == pytest.approx(p_qos_ref)
    assert cov_prob * qos_prob == pytest.approx(p_qos_ref * p_cov_ref)


def test_expected_scs_next_calls_mc_once(cfg, rng, monkeypatch):
    call_count = {"n": 0}
    original = scs_module.mc_coverage_prob

    def wrapped_mc(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scs_module, "mc_coverage_prob", wrapped_mc)
    p = ProviderRecord(
        service_id="p0",
        timestamp=0,
        response_time_ms=1,
        throughput_kbps=1,
        cost=0.0,
        coords=(0.0, 0.0),
        qos="Low",
        qos_prob=0.2,
        qos_volatility=0.0,
    )
    c = ConsumerRecord(
        service_id="c0",
        timestamp=0,
        response_time_ms=1,
        throughput_kbps=1,
        cost=0.0,
        coords=(5.0, 0.0),
        qos=None,
        qos_prob=0.0,
        qos_volatility=0.0,
    )
    assign = [0]
    prev_assign = [0]
    expected_scs_next(assign, ([p], [c]), prev_assign, cfg, cfg.scs, rng)
    assert call_count["n"] == 1


def _make_records():
    p0 = ProviderRecord(
        service_id="p0",
        timestamp=0,
        response_time_ms=0.0,
        throughput_kbps=0.0,
        cost=0.0,
        coords=(0.0, 0.0),
        qos="High",
        qos_prob=1.0,
        qos_volatility=0.0,
    )
    p1 = ProviderRecord(
        service_id="p1",
        timestamp=0,
        response_time_ms=0.0,
        throughput_kbps=0.0,
        cost=0.0,
        coords=(0.0, 0.0),
        qos="High",
        qos_prob=1.0,
        qos_volatility=0.0,
    )
    c0 = ConsumerRecord(
        service_id="c0",
        timestamp=0,
        response_time_ms=0.0,
        throughput_kbps=0.0,
        cost=0.0,
        coords=(0.0, 0.0),
        qos=None,
        qos_prob=0.0,
        qos_volatility=0.0,
    )
    c1 = ConsumerRecord(
        service_id="c1",
        timestamp=0,
        response_time_ms=0.0,
        throughput_kbps=0.0,
        cost=0.0,
        coords=(0.0, 0.0),
        qos=None,
        qos_prob=0.0,
        qos_volatility=0.0,
    )
    return [p0, p1], [c0, c1]


def test_scs_continuity_prev_assign_types(cfg):
    prods, cons = _make_records()
    assign = [0, 1]
    _, comps = scs(assign, (prods, cons), [0], cfg, cfg.scs)
    assert comps.continuity == pytest.approx(0.5)
    _, comps = scs(assign, (prods, cons), {0: 0}, cfg, cfg.scs)
    assert comps.continuity == pytest.approx(0.5)
    _, comps = scs(assign, (prods, cons), {0: 0, 1: 1}, cfg, cfg.scs)
    assert comps.continuity == pytest.approx(1.0)


def test_expected_scs_next_continuity_prev_assign_types(cfg, rng):
    prods, cons = _make_records()
    assign = [0, 1]
    _, comps = expected_scs_next(assign, (prods, cons), [0], cfg, cfg.scs, rng, mc_rollouts=0)
    assert comps.continuity == pytest.approx(0.5)
    _, comps = expected_scs_next(assign, (prods, cons), {0: 0}, cfg, cfg.scs, rng, mc_rollouts=0)
    assert comps.continuity == pytest.approx(0.5)
    _, comps = expected_scs_next(assign, (prods, cons), {0: 0, 1: 1}, cfg, cfg.scs, rng, mc_rollouts=0)
    assert comps.continuity == pytest.approx(1.0)
