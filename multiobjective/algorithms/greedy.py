from typing import Tuple, List
import numpy as np
from ..types import ErrorType, ProviderRecord, ConsumerRecord
from ..config import Config, coverage_radius
from ..rng import RNGPool
from ..simulation import euclidean_distance
from ..qos import reg_err
from ..streaks import StreakTracker
from ..indicators import MetricsRecorder
from ..defaults import OU_PARAMS_DEFAULT
# and, if you need the helper:
from ..metrics.scs import blended_error


def greedy_run(cfg: Config, rng_pool: RNGPool, records: dict, cost_per: dict,
               err_type: ErrorType, metrics: MetricsRecorder,
               streaks: StreakTracker, norm_fn,
               transition_matrix: dict | None = None):
    errors, costs, stds = [], [], []
    radius = coverage_radius(cfg)
    for t in range(cfg.num_times):
        scs_rng = rng_pool.for_time("scs", t)

        prods, cons = records[t]
        curr_max, curr_min = max(cost_per[f"{t}"]), min(cost_per[f"{t}"])
        def cost_norm(x): return (x - curr_min) / (curr_max - curr_min + 1e-12)

        matched = []
        for c in cons:
            scores, idxs = [], []
            for i, p in enumerate(prods):
                if euclidean_distance(p, c) > radius:
                    continue
                r = p.qos_prob
                v = p.qos_volatility
            
                
                err_for_score = blended_error(
                    err_type, p, c, t, cfg, norm_fn, scs_rng,
                    ou_params=OU_PARAMS_DEFAULT,
                    transition_matrix=transition_matrix,
                )
                score = (
                    err_for_score
                    * cost_norm(p.cost)
                    * (1.0 + cfg.lambda_vol * v)
                    / (max(r, 1e-6) ** cfg.gamma_qos)
                )

                scores.append(score); idxs.append(i)
            if not idxs:
                raise RuntimeError(
                    f"No providers within coverage for c={c.service_id} at t={t}"
                )
            best_i = idxs[int(np.argmin(scores))]
            p = prods[best_i]
            # streak continuity bookkeeping
            streaks.update(t, c.service_id, p.service_id)
            e = blended_error(
                err_type, p, c, t, cfg, norm_fn, scs_rng,
                ou_params=OU_PARAMS_DEFAULT,
                transition_matrix=transition_matrix,
            )

            matched.append((e, cost_norm(p.cost)))

        avg_err = float(np.mean([m[0] for m in matched])); avg_cost=float(np.mean([m[1] for m in matched]))
        std_err = float(np.std([m[0] for m in matched])) if len(matched)>1 else 0.0
        errors.append(avg_err); costs.append(avg_cost); stds.append(std_err)
        metrics.record("greedy", err_type, t, [(avg_err, avg_cost)])
    return errors, costs, stds
