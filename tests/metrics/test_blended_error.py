import pytest

from multiobjective.metrics.scs import blended_error, SCSConfig, OUParams
from multiobjective.types import ProviderRecord, ConsumerRecord


def test_blended_error_returns_base_when_disabled(cfg, rng):
    cfg.scs = SCSConfig(enabled=False, weight=0.5, mc_samples=1)
    base = 0.3

    def norm_fn(kind, err, t):
        return base

    p = ProviderRecord(
        service_id="p0",
        timestamp=0,
        response_time_ms=1,
        throughput_kbps=1,
        cost=0.0,
        coords=(0.0, 0.0),
        qos=None,
        qos_prob=0.0,
        qos_volatility=0.0,
    )
    c = ConsumerRecord(
        service_id="c0",
        timestamp=0,
        response_time_ms=1,
        throughput_kbps=1,
        cost=0.0,
        coords=(1.0, 0.0),
        qos=None,
        qos_prob=0.0,
        qos_volatility=0.0,
    )
    result = blended_error("rel", p, c, 0, cfg, norm_fn, rng, OUParams(0.0, 0.0, 1.0))
    assert result == base


def test_blended_error_weights_when_enabled(cfg, rng):
    cfg.scs = SCSConfig(enabled=True, weight=0.5, mc_samples=1)
    base = 0.2

    def norm_fn(kind, err, t):
        return base

    p = ProviderRecord(
        service_id="p0",
        timestamp=0,
        response_time_ms=1,
        throughput_kbps=1,
        cost=0.0,
        coords=(0.0, 0.0),
        qos="High",
        qos_prob=0.0,
        qos_volatility=0.0,
    )
    c = ConsumerRecord(
        service_id="c0",
        timestamp=0,
        response_time_ms=1,
        throughput_kbps=1,
        cost=0.0,
        coords=(0.0, 0.0),
        qos=None,
        qos_prob=0.0,
        qos_volatility=0.0,
    )
    ou = OUParams(theta=0.0, sigma=0.0, delta_t=1.0)
    tm = {"High": {"High": 1.0}}
    result = blended_error(
        "rel",
        p,
        c,
        0,
        cfg,
        norm_fn,
        rng,
        ou_params=ou,
        transition_matrix=tm,
        mc_rollouts=1,
    )
    assert result == pytest.approx((1 - cfg.scs.weight) * base)
