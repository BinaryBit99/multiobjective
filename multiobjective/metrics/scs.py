from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from ..config import Config, coverage_radius
from ..qos import reg_err
from ..simulation import euclidean_distance
from ..types import ProviderRecord, ConsumerRecord


@dataclass
class SCSConfig:
    """Configuration for SCS look-ahead."""
    enabled: bool = True
    weight: float = 0.40
    mc_samples: int = 96


@dataclass
class SCSComponents:
    """Aggregated components of the SCS metric."""
    continuity: float = 0.0
    coverage: float = 0.0
    qos: float = 0.0
    value: float = 0.0

@dataclass(frozen=True)
class OUParams:
    theta: float      # mean reversion
    sigma: float      # diffusion
    delta_t: float    # step

def _ou_step(x_t: np.ndarray, mu: np.ndarray, ou: OUParams, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(size=x_t.shape)
    return x_t + ou.theta*(mu - x_t)*ou.delta_t + ou.sigma*np.sqrt(ou.delta_t)*noise

def _clip_box(x: np.ndarray, high: Tuple[float,float]) -> np.ndarray:
    lo = np.array([0.0, 0.0], dtype=float)
    hi = np.array(high, dtype=float)
    return np.clip(x, lo, hi)

def mc_coverage_prob(
    p_coords: Tuple[float,float],
    c_coords: Tuple[float,float],
    space_size: Tuple[float,float],
    radius: float,
    ou: OUParams,
    rng: np.random.Generator,
    mean_p: Optional[np.ndarray] = None,
    mean_c: Optional[np.ndarray] = None,
    K: int = 128,
) -> float:
    """Monte Carlo estimate of P(||P_{t+1}-C_{t+1}|| <= radius) for one OU step."""
    p = np.asarray(p_coords, dtype=float)
    c = np.asarray(c_coords, dtype=float)
    center = np.array(space_size, dtype=float) / 2.0
    mu_p = center if mean_p is None else np.asarray(mean_p, dtype=float)
    mu_c = center if mean_c is None else np.asarray(mean_c, dtype=float)
    if K <= 0:
        return 0.0

    # Generate provider and consumer perturbations in one vectorized call. The
    # flattened layout preserves the original random-number ordering where each
    # provider sample is immediately followed by its paired consumer sample.
    noise = rng.standard_normal((K, 4))
    noise_p = noise[:, :2]
    noise_c = noise[:, 2:]

    drift_p = p + ou.theta * (mu_p - p) * ou.delta_t
    drift_c = c + ou.theta * (mu_c - c) * ou.delta_t
    scale = ou.sigma * np.sqrt(ou.delta_t)
    p1 = _clip_box(drift_p + scale * noise_p, space_size)
    c1 = _clip_box(drift_c + scale * noise_c, space_size)

    dists = np.linalg.norm(p1 - c1, axis=1)
    return np.count_nonzero(dists <= radius) / K

def qos_success_prob(
    provider_qos_now: Optional[str],
    transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    fallback_prob: float = 0.5,
) -> float:
    """P(QoS acceptable at t+1). If Markov row is known, use it; else fallback."""
    if provider_qos_now and transition_matrix and provider_qos_now in transition_matrix:
        row = transition_matrix[provider_qos_now]
        return float(row.get("Medium", 0.0) + row.get("High", 0.0))
    return float(fallback_prob)

def expected_pair_scs_tplus1(
    provider_record: ProviderRecord,
    consumer_record: ConsumerRecord,
    space_size: Tuple[float, float],
    radius: float,
    ou: OUParams,
    rng: np.random.Generator,
    transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    mc_rollouts: int = 128,
) -> Tuple[float, float]:
    """Estimate next-step coverage and QoS success probabilities.

    The caller can combine these probabilities to form the expected
    service-continuity success for the pair without re-running the Monte
    Carlo coverage estimate.
    """
    p_qos_next = qos_success_prob(
        provider_qos_now=provider_record.qos,
        transition_matrix=transition_matrix,
        fallback_prob=provider_record.qos_prob,
    )

    p_cov_next = mc_coverage_prob(
        p_coords=provider_record.coords,
        c_coords=consumer_record.coords,
        space_size=space_size,
        radius=radius,
        ou=ou,
        rng=rng,
        K=mc_rollouts,
    )
    return p_cov_next, p_qos_next

def blended_error(
    err_type: str,
    p: ProviderRecord,
    c: ConsumerRecord,
    t: int,
    cfg: Config,
    norm_fn,                   # your existing norm_err(kind, err, t)
    rng: np.random.Generator,  # per-time RNG, e.g. rng_pool.for_time("scs", t)
    ou_params: Optional[OUParams] = None,
    transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    mc_rollouts: Optional[int] = None,
) -> float:
    """
    Combine current normalized error with (1 - E[SCS_{t+1}]):
        err' = (1-w)*err_now + w*(1 - E[SCS_{t+1}])
    """
    base = norm_fn(err_type, reg_err(p, c, err_type), t)
    scs_cfg = cfg.scs
    # If SCS is disabled or records lack QoS information, return base error.
    if not scs_cfg.enabled or isinstance(p, dict) or isinstance(c, dict):
        return base

    ou = ou_params or OUParams(cfg.ou_theta, cfg.ou_sigma, cfg.delta_t)
    radius = coverage_radius(cfg)
    samples = mc_rollouts if mc_rollouts is not None else scs_cfg.mc_samples

    cov_prob, qos_prob = expected_pair_scs_tplus1(
        provider_record=p,
        consumer_record=c,
        space_size=cfg.space_size,
        radius=radius,
        ou=ou,
        rng=rng,
        transition_matrix=transition_matrix,
        mc_rollouts=samples,
    )
    e_scs = cov_prob * qos_prob
    w = scs_cfg.weight
    return (1.0 - w) * base + w * (1.0 - e_scs)


def scs(
    assign: list[int],
    pc: Tuple[list[ProviderRecord], list[ConsumerRecord]],
    prev_assign: Optional[list[int]],
    cfg: Config,
    scs_cfg: SCSConfig,
) -> Tuple[float, SCSComponents]:
    prods, cons = pc
    num_c = len(cons)
    if num_c == 0:
        return 0.0, SCSComponents()

    radius = coverage_radius(cfg)
    cont_total = cov_total = qos_total = success_total = 0.0

    for ci, c in enumerate(cons):
        pi = assign[ci]
        p = prods[pi]

        cont = float(prev_assign is not None and prev_assign[ci] == pi)
        cov = float(euclidean_distance(p, c) <= radius)
        qos = float(p.qos in {"Medium", "High"})

        cont_total += cont
        cov_total += cov
        qos_total += qos
        success_total += cont * cov * qos

    cont_avg = cont_total / num_c
    cov_avg = cov_total / num_c
    qos_avg = qos_total / num_c
    value = success_total / num_c
    comps = SCSComponents(continuity=cont_avg, coverage=cov_avg, qos=qos_avg, value=value)
    return value, comps


def expected_scs_next(
    assign: list[int],
    pc: Tuple[list[ProviderRecord], list[ConsumerRecord]],
    prev_assign: Optional[list[int]],
    cfg: Config,
    scs_cfg: SCSConfig,
    rng: np.random.Generator,
    transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    mc_rollouts: Optional[int] = None,
) -> Tuple[float, SCSComponents]:
    prods, cons = pc
    num_c = len(cons)
    if num_c == 0:
        return 0.0, SCSComponents()

    radius = coverage_radius(cfg)
    ou = OUParams(cfg.ou_theta, cfg.ou_sigma, cfg.delta_t)
    samples = mc_rollouts if mc_rollouts is not None else scs_cfg.mc_samples

    cont_total = cov_total = qos_total = success_total = 0.0

    for ci, c in enumerate(cons):
        pi = assign[ci]
        p = prods[pi]

        cont = float(prev_assign is not None and prev_assign[ci] == pi)
        cov_prob, qos_prob = expected_pair_scs_tplus1(
            provider_record=p,
            consumer_record=c,
            space_size=cfg.space_size,
            radius=radius,
            ou=ou,
            rng=rng,
            transition_matrix=transition_matrix,
            mc_rollouts=samples,
        )

        cont_total += cont
        cov_total += cov_prob
        qos_total += qos_prob
        success_total += cont * cov_prob * qos_prob

    cont_avg = cont_total / num_c
    cov_avg = cov_total / num_c
    qos_avg = qos_total / num_c
    value = success_total / num_c
    comps = SCSComponents(continuity=cont_avg, coverage=cov_avg, qos=qos_avg, value=value)
    return value, comps
