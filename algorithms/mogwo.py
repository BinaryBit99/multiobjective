import numpy as np
from ..config import Config
from ..rng import RNGPool
from ..types import ErrorType, ProviderRecord, ConsumerRecord
from ..indicators import MetricsRecorder
from ..pareto import crowding_distance

from ..defaults import OU_PARAMS_DEFAULT
# and, if you need the helper:
from ..metrics.scs import blended_error





def _dominates(a,b): return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def _fast_nds(objs):
    n=len(objs)
    if n==0: return [[]]
    domc=[0]*n; dom=[[] for _ in range(n)]; fronts=[[]]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if _dominates(objs[i],objs[j]): dom[i].append(j)
            elif _dominates(objs[j],objs[i]): domc[i]+=1
        if domc[i]==0: fronts[0].append(i)
    cur=0
    while cur<len(fronts) and fronts[cur]:
        nxt=[]
        for i in fronts[cur]:
            for j in dom[i]:
                domc[j]-=1
                if domc[j]==0: nxt.append(j)
        if nxt: fronts.append(nxt)
        cur+=1
    return [f for f in fronts if f]

def run_mogwo(cfg: Config, rng_pool: RNGPool, records: dict, cost_per: dict,
              err_type: ErrorType, metrics: MetricsRecorder, norm_fn,
              transition_matrix: dict | None = None):
    errors, costs, stds = [], [], []
    for t in range(cfg.num_times):
        rng = rng_pool.for_("gwo", t)
        scs_rng = rng_pool.for_("scs", t)
        prods: list[ProviderRecord]; cons: list[ConsumerRecord]
        prods, cons = records[t]
        D, P = len(cons), len(prods)
        q = np.zeros((P,D)); c = np.zeros((P,D))
        curr_max, curr_min = max(cost_per[f"{t}"]), min(cost_per[f"{t}"])
        for p in range(P):
            for j, cn in enumerate(cons):
                q[p,j] = blended_error(
                    err_type,
                    prods[p], cn, t,
                    cfg, norm_fn, scs_rng,
                    ou_params=OU_PARAMS_DEFAULT,
                    transition_matrix=transition_matrix,
                )
                r = prods[p].qos_prob
                v = prods[p].qos_volatility
                base = (prods[p].cost - curr_min) / (curr_max - curr_min + 1e-12)
                c[p, j] = (
                    base * (1.0 + cfg.lambda_vol * v) / (max(r, 1e-6) ** cfg.gamma_qos)
                )

        wolves = rng.uniform(0, P-1, (cfg.gwo.wolf_size, D)).astype(np.float32)
        archive = []
        for it in range(cfg.gwo.max_iters):
            disc = np.clip(np.round(wolves).astype(int), 0, P-1)
            rows = disc; cols = np.tile(np.arange(D), (cfg.gwo.wolf_size, 1))
            errs = q[rows, cols]; costs_m = c[rows, cols]
            objs = list(zip(errs.mean(axis=1).tolist(), costs_m.mean(axis=1).tolist()))
            sols = list(zip(wolves.copy(), objs))
            all_sols = archive + sols; all_objs = [o for _,o in all_sols]
            fronts = _fast_nds(all_objs)
            if fronts:
                first = [all_sols[i] for i in fronts[0]]
                if len(first) <= cfg.gwo.archive_size:
                    archive = first
                else:
                    cd = crowding_distance([o for _,o in first])
                    top = np.argpartition(-np.array(cd), cfg.gwo.archive_size)[:cfg.gwo.archive_size]
                    archive = [first[i] for i in top]

            if archive:
                arr_pos, arr_obj = zip(*archive)
                cd = crowding_distance(list(arr_obj))
                idx = np.argsort(cd)[::-1]
                alpha, beta, delta = arr_pos[idx[0]], arr_pos[idx[1]], arr_pos[idx[2]]
            else:
                alpha = beta = delta = wolves[0]

            a = 2 - 2*it/max(1,cfg.gwo.max_iters)
            r = lambda: rng.random(D)
            upd = []
            for i in range(cfg.gwo.wolf_size):
                A1, C1 = 2*a*r() - a, 2*r()
                A2, C2 = 2*a*r() - a, 2*r()
                A3, C3 = 2*a*r() - a, 2*r()
                X1 = alpha - A1*np.abs(C1*alpha - wolves[i])
                X2 = beta  - A2*np.abs(C2*beta  - wolves[i])
                X3 = delta - A3*np.abs(C3*delta - wolves[i])
                upd.append(np.clip((X1+X2+X3)/3, 0, P-1))
            wolves = np.array(upd, dtype=np.float32)

        if archive:
            objs = [o for _,o in archive]
            metrics.record("mogwo", err_type, t, [objs[i] for i in _fast_nds(objs)[0]])
            arr = np.array(objs); errors.append(float(arr[:,0].mean())); costs.append(float(arr[:,1].mean()))
            stds.append(float(arr[:,0].std()) if len(arr)>1 else 0.0)
        else:
            metrics.record("mogwo", err_type, t, [])
            errors.append(0.0); costs.append(0.0); stds.append(0.0)
    return errors, costs, stds
