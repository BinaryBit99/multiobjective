from typing import Protocol, List, Tuple, Callable, Optional
import numpy as np, math
from ..types import ErrorType
from ..qos import reg_err
from ..config import Config
from ..rng import RNGPool

from ..config import Config
from ..rng import RNGPool
from ..metrics.scs import blended_error
from ..defaults import OU_PARAMS_DEFAULT


class Algorithm(Protocol):
    def run(self, cfg: Config, rng_pool: RNGPool, records: dict,
            cost_per: dict, err_type: ErrorType, ctx: dict) -> tuple[list[float], list[float], list[float]]: ...

class Individual:
    def __init__(self, genes: List[int]):
        self.genes = genes[:]  # provider indices
        self.error: float | None = None
        self.cost: float | None  = None
        self.rank = float('inf')
        self.crowding = 0.0

    def evaluate(
        self,
        prods,
        cons,
        err_type,
        norm_fn,
        t,
        gamma_qos,
        lambda_vol,
        cfg: Config,
        rng_pool: RNGPool,
        transition_matrix: dict | None = None,
        ):
        errs, costs = [], []
        for i, c in enumerate(cons):
            p = prods[self.genes[i]]
            e_raw = reg_err(p, c, err_type)
            errs.append(norm_fn(err_type, e_raw, t))
            # one per-time RNG stream for SCS lookahead
            scs_rng = rng_pool.for_("scs", t)

            e = blended_error(
                err_type,
                p, c, t,
                cfg,            # access to space, radius, weights, etc.
                norm_fn,        # your existing normalizer
                scs_rng,        # RNG for MC
                ou_params=OU_PARAMS_DEFAULT,
                transition_matrix=transition_matrix,
            )
            errs.append(e)

            r = p.get("qos_prob", 0.5)
            v = p.get("qos_volatility", 0.0)
            costs.append(ctx_norm_cost(p["cost"], t, norm_fn) * (1.0 + lambda_vol*v) / (max(r,1e-6)**gamma_qos))
        self.error = float(np.mean(errs))
        self.cost  = float(np.mean(costs))

def ctx_norm_cost(cost, t, norm_fn):
    # retrieves min/max from norm_fn context (set in experiment)
    return norm_fn("__cost__", cost, t)  # special kind

def fast_non_dominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    N = len(pop)
    S = {i: [] for i in range(N)}
    n = {i: 0 for i in range(N)}
    fronts_idx: List[List[int]] = [[]]
    for p in range(N):
        for q in range(N):
            if p==q: continue
            better_p = ((pop[p].error <= pop[q].error and pop[p].cost <= pop[q].cost) and
                        (pop[p].error < pop[q].error or pop[p].cost < pop[q].cost))
            better_q = ((pop[q].error <= pop[p].error and pop[q].cost <= pop[p].cost) and
                        (pop[q].error < pop[p].error or pop[q].cost < pop[p].cost))
            if better_p: S[p].append(q)
            elif better_q: n[p] += 1
        if n[p]==0: fronts_idx[0].append(p)
    i=0
    while fronts_idx[i]:
        nxt=[]
        for p_idx in fronts_idx[i]:
            for q_idx in S[p_idx]:
                n[q_idx]-=1
                if n[q_idx]==0: nxt.append(q_idx)
        i+=1; fronts_idx.append(nxt)
    if not fronts_idx[-1]: fronts_idx.pop()
    return [[pop[idx] for idx in fr] for fr in fronts_idx]

def calculate_crowding_distance(front: List[Individual]) -> list[tuple[Individual,float]]:
    import numpy as np
    n = len(front)
    if n==0: return []
    if n==1: return [(front[0], float("inf"))]
    errs = np.array([ind.error for ind in front]); costs = np.array([ind.cost for ind in front])
    err_idx = np.argsort(errs); cost_idx = np.argsort(costs)
    dist = np.zeros(n); dist[err_idx[[0,-1]]] = np.inf; dist[cost_idx[[0,-1]]] = np.inf
    if n>2:
        er = errs[err_idx[-1]] - errs[err_idx[0]] + 1e-12
        cr = costs[cost_idx[-1]] - costs[cost_idx[0]] + 1e-12
        dist[err_idx[1:-1]] += (errs[err_idx[2:]] - errs[err_idx[:-2]])/er
        dist[cost_idx[1:-1]] += (costs[cost_idx[2:]] - costs[cost_idx[:-2]])/cr
    pairs = list(zip(front, dist))
    pairs.sort(key=lambda x: (math.isinf(x[1]), x[1]), reverse=True)
    return pairs

def tournament_select(pop: List[Individual], k: int, rng) -> Individual:
    idx = rng.choice(len(pop), size=k, replace=False)
    contenders = [pop[int(i)] for i in idx]
    contenders.sort(key=lambda ind: (ind.rank, -ind.crowding))
    return contenders[0]
