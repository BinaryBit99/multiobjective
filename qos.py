import math, numpy as np
from typing import Literal
from .types import ProviderRecord, ConsumerRecord

QoSState = Literal["Low","Medium","High"]

def classify_qos(resp_norm: float, thrp_norm: float, alpha: float=0.5) -> QoSState:
    score = alpha*resp_norm + (1-alpha)*(1-thrp_norm)
    if score >= 0.66: return "Low"
    if score >= 0.33: return "Medium"
    return "High"

def smooth_qos(prev_qos: QoSState, inferred_qos: QoSState, transition_matrix: dict, beta: float, rng) -> QoSState:
    prob_next = transition_matrix[prev_qos].copy()
    prob_next[inferred_qos] = prob_next.get(inferred_qos,0.0) + beta
    states = list(prob_next.keys())
    probs  = np.array([prob_next[s] for s in states], dtype=float)
    probs /= probs.sum() if probs.sum()>0 else 1.0
    return rng.choice(states, p=probs)

def reg_err(p: ProviderRecord, c: ConsumerRecord, kind: str) -> float:
    if kind == "tp":
        prov, req = p.throughput_kbps, c.throughput_kbps
        if prov < req:
            return (req - prov) / (req + 1e-12)
        if req < prov:
            x = max(0.005 * ((prov - req) / (req + 1e-12)), 1e-6)
            return abs(0.5 * math.log(x))
        return 0.0
    elif kind == "res":
        prov, req = p.response_time_ms, c.response_time_ms
        if prov < req:
            x = max(0.005 * ((req - prov) / (req + 1e-12)), 1e-6)
            return abs(0.5 * math.log(x))
        if req < prov:
            return (prov - req) / (req + 1e-12)
        return 0.0
    else:
        # relative
        den = (c.response_time_ms + c.throughput_kbps) or 1.0
        return abs(1 - ((p.response_time_ms + p.throughput_kbps) / den))
