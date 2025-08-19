# metrics/scs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List, Sequence
import numpy as np

from ..config import Config, coverage_radius
from ..qos import reg_err
from ..rng import RNGPool

@dataclass(frozen=True)
class OUParams:
    theta: float      # mean reversion
    sigma: float      # diffusion
    delta_t: float    # step


@dataclass
class SCSConfig:
    """Configuration for SCS related calculations."""
    enabled: bool = True
    weight: float = 0.5
    mc_samples: int = 128


@dataclass
class SCSComponents:
    """Breakdown of SCS sub-metrics."""
    coverage: float
    continuity: float
    qos: float

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

    hits = 0
    for _ in range(K):
        p1 = _clip_box(_ou_step(p, mu_p, ou, rng), space_size)
        c1 = _clip_box(_ou_step(c, mu_c, ou, rng), space_size)
        if np.linalg.norm(p1 - c1) <= radius:
            hits += 1
    return hits / max(K, 1)

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
    provider_record: dict,
    consumer_record: dict,
    space_size: Tuple[float,float],
    radius: float,
    ou: OUParams,
    rng: np.random.Generator,
    transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    mc_rollouts: int = 128,
) -> float:
    """
    E[SCS_{t+1}] ≈ P(within radius at t+1) * P(QoS acceptable at t+1)
    """
    p_qos_next = qos_success_prob(
        provider_qos_now = provider_record.get("qos"),
        transition_matrix = transition_matrix,
        fallback_prob = provider_record.get("qos_prob", 0.5),
    )
    p_cov_next = mc_coverage_prob(
        p_coords = tuple(provider_record["coords"]),
        c_coords = tuple(consumer_record["coords"]),
        space_size = space_size,
        radius = radius,
        ou = ou,
        rng = rng,
        K = mc_rollouts,
    )
    return p_cov_next * p_qos_next

def blended_error(
    err_type: str,
    p: dict,
    c: dict,
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
    scs_cfg = getattr(cfg, "scs", None)
    if not scs_cfg or not scs_cfg.enabled:
        return base

    ou = ou_params or OUParams(cfg.ou_theta, cfg.ou_sigma, cfg.delta_t)
    radius = coverage_radius(cfg)
    samples = mc_rollouts if mc_rollouts is not None else scs_cfg.mc_samples

    e_scs = expected_pair_scs_tplus1(
        provider_record=p,
        consumer_record=c,
        space_size=cfg.space_size,
        radius=radius,
        ou=ou,
        rng=rng,
        transition_matrix=transition_matrix,
        mc_rollouts=samples,
    )
    w = scs_cfg.weight
    return (1.0 - w) * base + w * (1.0 - e_scs)


def scs(
    assign: Sequence[int],
    records: Tuple[Sequence[dict], Sequence[dict]],
    prev_assign: Optional[Dict[str, str]],
    cfg: Config,
    scs_cfg: SCSConfig,
) -> Tuple[float, SCSComponents]:
    """Compute current SCS for a set of assignments."""
    prods, cons = records
    n = len(cons) or 1
    radius = coverage_radius(cfg)
    cov_hits = cont_hits = qos_hits = 0
    for ci, p_idx in enumerate(assign):
        p = prods[p_idx]
        c = cons[ci]
        (px, py), (cx, cy) = p["coords"], c["coords"]
        if ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 <= radius:
            cov_hits += 1
        if prev_assign and prev_assign.get(c["service_id"]) == p["service_id"]:
            cont_hits += 1
        if p.get("qos") in ("Medium", "High"):
            qos_hits += 1
    coverage = cov_hits / n
    continuity = (cont_hits / n) if prev_assign else 1.0
    qos = qos_hits / n
    score = coverage * qos * continuity
    return score, SCSComponents(coverage, continuity, qos)


def expected_scs_next(
    assign: Sequence[int],
    records: Tuple[Sequence[dict], Sequence[dict]],
    prev_assign: Optional[Dict[str, str]],
    cfg: Config,
    scs_cfg: SCSConfig,
    rng_pool: RNGPool,
    transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[float, SCSComponents]:
    """Estimate mean SCS at the next time step."""
    prods, cons = records
    n = len(cons) or 1
    radius = coverage_radius(cfg)
    ou = OUParams(cfg.ou_theta, cfg.ou_sigma, cfg.delta_t)
    cov_probs: List[float] = []
    qos_probs: List[float] = []
    for ci, p_idx in enumerate(assign):
        p = prods[p_idx]
        c = cons[ci]
        cov_probs.append(
            mc_coverage_prob(
                p_coords=tuple(p["coords"]),
                c_coords=tuple(c["coords"]),
                space_size=cfg.space_size,
                radius=radius,
                ou=ou,
                rng=rng_pool.global_,
                K=scs_cfg.mc_samples,
            )
        )
        qos_probs.append(
            qos_success_prob(
                provider_qos_now=p.get("qos"),
                transition_matrix=transition_matrix,
                fallback_prob=p.get("qos_prob", 0.5),
            )
        )
    coverage = float(np.mean(cov_probs)) if cov_probs else 0.0
    qos = float(np.mean(qos_probs)) if qos_probs else 0.0
    cont_hits = 0
    if prev_assign:
        for ci, p_idx in enumerate(assign):
            c = cons[ci]
            p = prods[p_idx]
            if prev_assign.get(c["service_id"]) == p["service_id"]:
                cont_hits += 1
    continuity = (cont_hits / n) if prev_assign else 1.0
    score = coverage * qos * continuity
    return score, SCSComponents(coverage, continuity, qos)
