import numpy as np
from .config import Config, coverage_radius
from .rng import RNGPool
from .data import RecordBuilder
from .indicators import MetricsRecorder
from .streaks import StreakTracker
from .metrics import SCSConfig, scs, expected_scs_next
from .qos import reg_err
from .types import ProviderRecord, ConsumerRecord
from .errors import CoverageError

def run_experiment(cfg: Config) -> dict:
    rng_pool = RNGPool(cfg.master_seed, cfg.num_times)
    records, cost_per_dict, T, _, _ = RecordBuilder(cfg, rng_pool)
    num_providers, num_consumers = cfg.num_providers, cfg.num_consumers

    # normalization bounds (per-time) for errors + cost
    per_time_bounds = {}
    rad = coverage_radius(cfg)
    for t in range(cfg.num_times):
        prods, cons = records[t]
        tp_errs, res_errs = [], []
        for c in cons:
            feasible = False
            for p in prods:
                if _within(p, c, rad):
                    feasible = True
                    tp_errs.append(reg_err(p, c, "tp"))
                    res_errs.append(reg_err(p, c, "res"))
            if not feasible:
                raise CoverageError(consumer_id=c.service_id, t=t, radius=rad)
        if not tp_errs or not res_errs:
            raise CoverageError(consumer_id="*", t=t, radius=rad)
        per_time_bounds[f"{t}"] = (
            max(tp_errs), max(res_errs), min(tp_errs), min(res_errs),
            min(cost_per_dict[f"{t}"]), max(cost_per_dict[f"{t}"])
        )

    def norm_err(kind, error, time):
        mx_tp, mx_res, mn_tp, mn_res, min_cost, max_cost = per_time_bounds[f"{time}"]
        if kind == "tp":
            return (error - mn_tp) / ((mx_tp - mn_tp) or 1e-12)
        if kind == "res":
            return (error - mn_res) / ((mx_res - mn_res) or 1e-12)
        if kind == "__cost__":
            return (error - min_cost) / ((max_cost - min_cost) or 1e-12)
        raise ValueError(kind)

    # trackers & metrics
    consumer_ids = [c.service_id for c in records[0][1]]
    streaks = StreakTracker(consumer_ids, cfg.num_times)
    metrics = MetricsRecorder(cfg.num_times)

    # run all algos
    from .algorithms import ALG_REGISTRY
    outputs = {}

    scs_cfg = (
        SCSConfig(**vars(cfg.scs)) if getattr(cfg, "scs", None) else SCSConfig()
    )

    for alg_name, fn in ALG_REGISTRY.items():
        series = {}
        for te in ["tp", "res"]:
            if alg_name == "greedy":
                errs, costs, stds = fn(
                    cfg,
                    rng_pool,
                    records,
                    cost_per_dict,
                    te,
                    metrics,
                    streaks,
                    norm_err,
                )
            else:
                errs, costs, stds = fn(
                    cfg,
                    rng_pool,
                    records,
                    cost_per_dict,
                    te,
                    metrics,
                    norm_err,
                )
            series.setdefault("errors", {})[te] = errs
            series.setdefault("costs", {})[te] = costs
            series.setdefault("stds", {})[te] = stds

            # SCS metrics per algorithm/error-type
            prev_assign = None
            actual, expected = [], []
            for t in range(cfg.num_times):
                prods, cons = records[t]
                assign = []
                for ci, c in enumerate(cons):
                    scores = []
                    for pi, p in enumerate(prods):
                        e = norm_err(te, reg_err(p, c, te), t)
                        scores.append((e + norm_err("__cost__", p.cost, t), pi))
                    assign.append(min(scores)[1])
                scs_rng = rng_pool.for_("scs", t)
                score, _ = scs(assign, (prods, cons), prev_assign, cfg, scs_cfg)
                mean_next, _ = expected_scs_next(
                    assign, (prods, cons), prev_assign, cfg, scs_cfg, scs_rng, T
                )
                actual.append(score)
                expected.append(mean_next)
                prev_assign = assign[:]
            series.setdefault("scs", {})[te] = {
                "actual": actual,
                "expected": expected,
            }

        outputs[alg_name] = series

    # compute indicators from logged fronts
    indicators = metrics.compute_all()

    return {
        "series": outputs,
        "indicators": indicators,
        "meta": {
            "num_providers": num_providers,
            "num_consumers": num_consumers,
            "transition_matrix": T,
        },
    }

def _within(p: ProviderRecord, c: ConsumerRecord, radius: float) -> bool:  # local
    (px, py), (cx, cy) = p.coords, c.coords
    dx = px - cx
    dy = py - cy
    return (dx * dx + dy * dy) ** 0.5 <= radius
