import numpy as np, time
from ..config import Config
from ..rng import RNGPool
from ..types import ErrorType, ProviderRecord, ConsumerRecord
from ..indicators import MetricsRecorder
from ..pareto import crowding_distance


# and, if you need the helper:


from ..metrics.scs import blended_error
from ..defaults import OU_PARAMS_DEFAULT

def _dominates(a, b):  # tuples
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def _fast_nds(objs):
    n=len(objs)
    if n==0: return [[]]
    domc=[0]*n; dom=[[] for _ in range(n)]; fronts=[[]]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if _dominates(objs[i], objs[j]): dom[i].append(j)
            elif _dominates(objs[j], objs[i]): domc[i]+=1
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

def run_mopso(cfg: Config, rng_pool: RNGPool, records: dict, cost_per: dict,
              err_type: ErrorType, metrics: MetricsRecorder, norm_fn,
              transition_matrix: dict | None = None):
    errors, costs, stds, times = [], [], [], []
    start = time.perf_counter()
    for t in range(cfg.num_times):
        rng = rng_pool.for_("pso", t)
        scs_rng = rng_pool.for_("scs", t)
        prods: list[ProviderRecord]; cons: list[ConsumerRecord]
        prods, cons = records[t]
        D, P = len(cons), len(prods)
        # QoS/Cost matrices
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

        # init swarm
        pos = rng.uniform(0, P-1, (cfg.pso.swarm_size, D)).astype(np.float32)
        vel = rng.uniform(-cfg.pso.v_max, cfg.pso.v_max, (cfg.pso.swarm_size, D)).astype(np.float32)
        pbest_pos = pos.copy()
        pbest_obj = np.full((cfg.pso.swarm_size,2), np.inf, dtype=np.float32)
        archive = []

        for it in range(cfg.pso.max_iterations):
            w = cfg.pso.w_max - (cfg.pso.w_max - cfg.pso.w_min) * (it / max(1, cfg.pso.max_iterations-1))
            disc = np.clip(np.round(pos).astype(int), 0, P-1)
            rows = disc; cols = np.tile(np.arange(D), (cfg.pso.swarm_size, 1))
            errs = q[rows, cols]; costs_mat = c[rows, cols]
            obj = np.stack((errs.mean(axis=1), costs_mat.mean(axis=1)), axis=1)

            # pbest update
            for i in range(cfg.pso.swarm_size):
                if _dominates(tuple(obj[i]), tuple(pbest_obj[i])):
                    pbest_obj[i] = obj[i]; pbest_pos[i] = pos[i].copy()

            # archive
            sols = [(pos[i].copy(), tuple(obj[i])) for i in range(cfg.pso.swarm_size)]
            all_sols = archive + sols
            all_objs = [o for _,o in all_sols]
            fronts = _fast_nds(all_objs)
            if fronts:
                first = [all_sols[i] for i in fronts[0]]
                if len(first) <= cfg.pso.archive_size:
                    archive = first
                else:
                    cd = crowding_distance([o for _,o in first])
                    top = np.argpartition(-np.array(cd), cfg.pso.archive_size)[:cfg.pso.archive_size]
                    archive = [first[i] for i in top]

            leaders = [pbest_pos[i] for i in range(min(3, len(pbest_pos)))] if not archive else [archive[0][0]]
            for i in range(cfg.pso.swarm_size):
                gbest = leaders[int(rng.integers(0, len(leaders)))]
                r1 = rng.random(D); r2 = rng.random(D)
                vel[i] = w*vel[i] + cfg.pso.c1*r1*(pbest_pos[i]-pos[i]) + cfg.pso.c2*r2*(gbest-pos[i])
                vel[i] = np.clip(vel[i], -cfg.pso.v_max, cfg.pso.v_max)
                pos[i] = np.clip(pos[i] + vel[i], 0, P-1)

        if archive:
            objs = [o for _,o in archive]; metrics.record("mopso", err_type, t, [objs[i] for i in _fast_nds(objs)[0]])
            arr = np.array(objs); errors.append(float(arr[:,0].mean())); costs.append(float(arr[:,1].mean()))
            stds.append(float(arr[:,0].std()) if len(arr)>1 else 0.0)
        else:
            metrics.record("mopso", err_type, t, [])
            errors.append(0.0); costs.append(0.0); stds.append(0.0)
        times.append(time.perf_counter() - start)
    return errors, costs, stds, times
