import numpy as np
from collections import defaultdict
from .pareto import pareto_prune

def _clip01_pair(p): return (min(max(p[0],0.0),1.0), min(max(p[1],0.0),1.0))

def hypervolume_2d(front, ref=(1.0,1.0)) -> float:
    if not front: return 0.0
    pts = pareto_prune([_clip01_pair(p) for p in front])
    if not pts: return 0.0
    pts = sorted(pts, key=lambda a: a[0], reverse=True)
    hv, prev_x, floor_y = 0.0, ref[0], ref[1]
    for x, y in pts:
        hv += max(prev_x-x, 0.0)*max(ref[1]-floor_y, 0.0)
        prev_x = x
        floor_y = min(floor_y, y)
    return hv

def igd(front, ref_set):
    if not ref_set: return float("nan")
    if not front:   return float("inf")
    def euclid(a,b): return float(np.linalg.norm(np.array(a)-np.array(b)))
    return float(np.mean([min(euclid(r,p) for p in front) for r in ref_set]))

def epsilon_additive(front, ref_set):
    if not ref_set: return float("nan")
    if not front:   return float("inf")
    eps = []
    for r in ref_set:
        eps.append(min(max(p[0]-r[0], p[1]-r[1]) for p in front))
    return float(max(0.0, max(eps)))

class MetricsRecorder:
    """
    Accumulate fronts per algo, error-type, time; compute HV/IGD/EPS.
    """
    def __init__(self, num_times: int):
        self.num_times = num_times
        self.front_log = defaultdict(lambda: {'tp': defaultdict(list), 'res': defaultdict(list)})

    def record(self, alg: str, err_type: str, t: int, objs: list[tuple[float,float]]):
        from .pareto import pareto_prune
        self.front_log[alg][err_type][t] = pareto_prune([_clip01_pair(p) for p in objs])

    def reference_set(self, err_type: str, t: int):
        union = []
        for alg in self.front_log.keys():
            union.extend(self.front_log[alg][err_type].get(t, []))
        return pareto_prune(union)

    def compute_all(self):
        metrics = defaultdict(lambda: {'tp': {}, 'res': {}})
        for te in ['tp','res']:
            for t in range(self.num_times):
                ref = self.reference_set(te, t)
                for alg in list(self.front_log.keys()):
                    front = self.front_log[alg][te].get(t, [])
                    HV  = hypervolume_2d(front, ref=(1.0,1.0))
                    IGD = igd(front, ref)
                    EPS = epsilon_additive(front, ref)
                    for name, val in [('HV',HV), ('IGD',IGD), ('EPS',EPS)]:
                        metrics[alg][te].setdefault(name, []).append(val)
        return metrics
