from typing import List, Tuple
import numpy as np

def dominates(a: Tuple[float,float], b: Tuple[float,float]) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def nondominated_indices(objs: List[Tuple[float,float]]) -> List[int]:
    n = len(objs)
    keep = [True]*n
    for i in range(n):
        for j in range(n):
            if i != j and dominates(objs[j], objs[i]):
                keep[i] = False
                break
    return [i for i, k in enumerate(keep) if k]

def pareto_prune(objs: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    idx = nondominated_indices(objs)
    return [objs[i] for i in idx]

def crowding_distance(objs: List[Tuple[float,float]]) -> list[float]:
    n = len(objs)
    if n == 0: return []
    if n == 1: return [float("inf")]
    e = np.array([o[0] for o in objs]); c = np.array([o[1] for o in objs])
    dist = np.zeros(n)
    eidx = np.argsort(e); cidx = np.argsort(c)
    dist[eidx[[0,-1]]] = float("inf"); dist[cidx[[0,-1]]] = float("inf")
    if n > 2:
        er = e[eidx[-1]] - e[eidx[0]] + 1e-12
        cr = c[cidx[-1]] - c[cidx[0]] + 1e-12
        dist[eidx[1:-1]] += (e[eidx[2:]] - e[eidx[:-2]])/er
        dist[cidx[1:-1]] += (c[cidx[2:]] - c[cidx[:-2]])/cr
    return dist.tolist()
