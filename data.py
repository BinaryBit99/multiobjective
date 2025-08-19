import numpy as np, pandas as pd, os
from typing import List, Tuple
from .config import Config
from .rng import RNGPool
from .simulation import build_trajectories
from .qos import classify_qos, smooth_qos

def _processing(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    mat = df.to_numpy()
    return mat[:, :-1] if mat.shape[1] == 5826 else mat

def load_qws_txt(path: str) -> Tuple[list[str], list[str]]:
    rs, tp = [], []
    with open(path, "r") as f:
        for line in f:
            fields = line.strip().split(",")
            if len(fields) >= 3:
                rs.append(fields[0]); tp.append(fields[2])
    return rs, tp

def RecordBuilder(cfg: Config, rng_pool: RNGPool):
    """
    Returns a callable build_all() that returns:
      - records: dict[t] -> (prods, cons) lists of dicts
      - cost_per_dict: dict[str t] -> list[float] costs of producers
      - transition_matrix: learned Markov T from raw provider QoS
    """
    num_providers, num_consumers = cfg.num_providers, cfg.num_consumers
    service_ids = [f"S{i+1}" for i in range(cfg.num_services)]
    providers = service_ids[:num_providers]
    consumers = service_ids[num_providers:]

    # try to load; if not, synthetic
    try:
        rt = _processing(os.getenv("RT_MATRIX", "rtMatrix.txt"))
        tp = _processing(os.getenv("TP_MATRIX", "tpMatrix.txt"))
    except Exception:
        rt = rng_pool.init.random((cfg.num_services, cfg.num_services))
        tp = rng_pool.init.random((cfg.num_services, cfg.num_services))

    df_rt, df_tp = pd.DataFrame(rt), pd.DataFrame(tp)
    service_rt_list = (df_rt.median(axis=0) * 1000).tolist()
    service_tp_list = df_tp.median(axis=0).tolist()

    rs_two, tp_two = service_rt_list, service_tp_list
    max_rt = max(float(x) for x in rs_two) if rs_two else 1.0
    max_tp = max(float(x) for x in tp_two) if tp_two else 1.0

    # initial QoS seeds + trajectories
    traj = build_trajectories(cfg, rng_pool, num_providers, num_consumers)
    provider_qos_raw = {i: ["Medium"] for i in range(num_providers)}
    T_counts = {s: {"Low":0,"Medium":0,"High":0} for s in ["Low","Medium","High"]}

    records = {}
    cost_per_dict = {}

    for t in range(cfg.num_times):
        rng = rng_pool.for_("init", t)
        k_rt = min(len(rs_two), cfg.num_services)
        k_tp = min(len(tp_two), cfg.num_services)
        rt_idx = rng.permutation(len(rs_two))[:k_rt]
        tp_idx = rng.permutation(len(tp_two))[:k_tp]

        prods, cons, costs = [], [], []
        coords = []
        for i in range(cfg.num_services):
            sid = f"p{i}" if i < num_providers else f"c{i - num_providers}"
            coords.append(traj[sid][t])

        for i, sid in enumerate(service_ids):
            ri = rt_idx[i % len(rt_idx)]; rj = tp_idx[i % len(tp_idx)]
            resp = float(rs_two[ri]); thrp = float(tp_two[rj])
            qos = None; qprob = 0.5; qvol = 0.0; cost = 0.0

            if i < num_providers:
                cost = float(rng.random())
                costs.append(cost)
                resp_norm = resp / max_rt if max_rt>0 else 0.5
                thrp_norm = thrp / max_tp if max_tp>0 else 0.5
                inferred = classify_qos(resp_norm, 1.0 - (1.0 - thrp_norm))  # same as classify(resp_norm, thrp_norm)
                prev = provider_qos_raw[i][-1] if t > 0 else "Medium"
                # temporary T: assume persistence; we’ll learn below
                provider_qos_raw[i].append(inferred)
                qos = inferred
                # heuristic reliability & volatility
                qprob = 0.7 if qos in ("Medium","High") else 0.3
                qvol  = 0.3 if qos == "Medium" else (0.1 if qos=="High" else 0.6)

            rec = {
                "service_id": sid, "timestamp": t,
                "response_time_ms": resp, "throughput_kbps": thrp,
                "cost": cost, "coords": coords[i].tolist(),
                "qos": qos, "qos_prob": qprob, "qos_volatility": qvol
            }
            (prods if i < num_providers else cons).append(rec)

        cost_per_dict[f"{t}"] = costs
        records[t] = (prods, cons)

    # learn transition matrix with Laplace = 1
    for seq in provider_qos_raw.values():
        for k in range(1, len(seq)):
            T_counts[seq[k-1]][seq[k]] += 1
    T = {}
    for prev, d in T_counts.items():
        tot = sum(v+1 for v in d.values())
        T[prev] = {s: (d[s]+1)/tot for s in d.keys()}

    return records, cost_per_dict, T, providers, consumers
