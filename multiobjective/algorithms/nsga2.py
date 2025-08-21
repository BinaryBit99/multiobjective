import numpy as np, math
from typing import List, Tuple
from .base import Individual, fast_non_dominated_sort, calculate_crowding_distance, tournament_select
from ..config import Config
from ..rng import RNGPool
from ..types import ErrorType
from ..simulation import euclidean_distance
from ..indicators import MetricsRecorder
from ..pareto import pareto_prune

# and, if you need the helper:
from ..metrics.scs import blended_error
from ..defaults import OU_PARAMS_DEFAULT



def run_nsga2(cfg: Config, rng_pool: RNGPool, records: dict, cost_per: dict,
              err_type: ErrorType, metrics: MetricsRecorder, norm_fn,
              transition_matrix: dict | None = None):
    pop_size = cfg.nsga.population_size
    best_fronts = []
    for t in range(cfg.num_times):
        rng = rng_pool.for_("nsga", t)
        prods, cons = records[t]
        D = len(cons)

        # seed population (feasible)
        pop: List[Individual] = []
        while len(pop) < pop_size:
            genes = []
            for i, c in enumerate(cons):
                valid = [j for j,p in enumerate(prods) if euclidean_distance(p,c) <= math.inf]  # radius handled in norm bounds already
                g = int(rng.choice(valid)) if valid else int(rng.integers(0, len(prods)))
                genes.append(g)
            ind = Individual(genes)
            ind.evaluate(prods, cons, err_type, norm_fn, t,
             cfg.gamma_qos, cfg.lambda_vol,
             cfg, rng_pool, transition_matrix)
            pop.append(ind)

        # rank/crowding before loop
        fronts = fast_non_dominated_sort(pop)
        for rank, front in enumerate(fronts):
            for ind, cd in calculate_crowding_distance(front):
                ind.rank = rank; ind.crowding = cd

        stagnant = 0; best = pareto_prune([(i.error, i.cost) for i in fronts[0]])
        best_score = np.mean([sum(p) for p in best]) if best else float("inf")

        for gen in range(1, cfg.nsga.max_generations + 1):
            pm = rng.uniform(cfg.nsga.mutation_prob_min, cfg.nsga.mutation_prob_max)
            pc = rng.uniform(cfg.nsga.crossover_prob_min, cfg.nsga.crossover_prob_max)
            off = []
            while len(off) < pop_size:
                p1 = tournament_select(pop, cfg.nsga.tournament_size, rng)
                p2 = tournament_select(pop, cfg.nsga.tournament_size, rng)
                c1, c2 = _sbx(p1, p2, cfg.nsga.crossover_eta, pc, rng, len(cons), len(prods))
                _eval(c1, prods, cons, err_type, norm_fn, t, cfg, rng_pool, transition_matrix)
                _eval(c2, prods, cons, err_type, norm_fn, t, cfg, rng_pool, transition_matrix)
                m1 = _poly_mut(c1, cfg.nsga.mutation_eta, pm, rng, len(prods), cons)
                _eval(m1, prods, cons, err_type, norm_fn, t, cfg, rng_pool, transition_matrix)
                off.append(m1)
                if len(off) < pop_size:
                    m2 = _poly_mut(c2, cfg.nsga.mutation_eta, pm, rng, len(prods), cons)
                    _eval(m2, prods, cons, err_type, norm_fn, t, cfg, rng_pool, transition_matrix)
                    off.append(m2)

            combined = pop + off
            fronts = fast_non_dominated_sort(combined)
            # assign rank/crowding
            new_pop = []
            for rank, fr in enumerate(fronts):
                cd_pairs = calculate_crowding_distance(fr)
                for ind, cd in cd_pairs: ind.rank = rank; ind.crowding = cd
                if len(new_pop) + len(fr) <= pop_size:
                    new_pop.extend(fr)
                else:
                    sorted_fr = sorted(fr, key=lambda i: i.crowding, reverse=True)
                    new_pop.extend(sorted_fr[:pop_size - len(new_pop)])
                    break
            pop = new_pop
            current_front = fast_non_dominated_sort(pop)[0]
            objs = [(i.error, i.cost) for i in current_front]
            score = np.mean([sum(p) for p in objs])
            if score < best_score:
                best_score = score; best = pareto_prune(objs); stagnant = 0
            else:
                stagnant += 1
                if stagnant >= cfg.nsga.patience: break

        metrics.record("nsga", err_type, t, best)
        best_fronts.append(best)

    # return mean series over time from best fronts
    mean_err = [np.mean([e for e,_ in bf]) if bf else 0.0 for bf in best_fronts]
    mean_cost= [np.mean([c for _,c in bf]) if bf else 0.0 for bf in best_fronts]
    std_err  = [np.std([e for e,_ in bf]) if len(bf)>1 else 0.0 for bf in best_fronts]
    return mean_err, mean_cost, std_err

def _sbx(p1, p2, eta, pc, rng, D, P):
    if rng.random() > pc:
        return _noisy_clone(p1, rng, P), _noisy_clone(p2, rng, P)
    c1, c2 = [], []
    for i in range(D):
        g1, g2 = p1.genes[i], p2.genes[i]
        if rng.random() <= 0.5 and abs(g1 - g2) > 1e-6:
            u = rng.random()
            beta = (2*u)**(1/(eta+1)) if u<=0.5 else (1/(2*(1-u)))**(1/(eta+1))
            v1 = 0.5*((1+beta)*g1 + (1-beta)*g2)
            v2 = 0.5*((1-beta)*g1 + (1+beta)*g2)
            c1.append(int(np.clip(round(v1), 0, P-1))); c2.append(int(np.clip(round(v2),0,P-1)))
        else:
            c1.append(g1); c2.append(g2)
    return Individual(c1), Individual(c2)

def _noisy_clone(ind, rng, P):
    g = [(gi if rng.random()>0.1 else int(rng.integers(0,P))) for gi in ind.genes]
    return Individual(g)

def _poly_mut(ind, eta, pm, rng, P, cons):
    if rng.random() > pm: return Individual(ind.genes[:])
    i = int(rng.integers(0, len(cons)))
    x = ind.genes[:]
    u = rng.random()
    delta = (2*u)**(1/(eta+1)) - 1 if u<0.5 else 1 - (2*(1-u))**(1/(eta+1))
    xn = int(np.clip(x[i] + round(delta*(P-1)), 0, P-1))
    x[i] = xn
    return Individual(x)


def _eval(ind, prods, cons, et, norm_fn, t, cfg: Config,
        rng_pool: RNGPool, transition_matrix: dict | None):

    ind.evaluate(prods, cons, et, norm_fn, t,
             cfg.gamma_qos, cfg.lambda_vol,
             cfg, rng_pool, transition_matrix)

    


