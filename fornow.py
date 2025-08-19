import pandas as pd
import numpy as np

import math
import matplotlib.pyplot as plt
from typing import List, Any, Tuple
from itertools import combinations
from collections import defaultdict

num_times      = 30
# General Parameters
SPATIAL_DISTRIBUTION = 'uniform'  # Options: 'uniform', 'clumped', 'random'
SPACE_SIZE = (100, 100)           # Size of 2D space, e.g., 100x100 units
NUM_CLUSTERS = 3                  # Only used for clumped
CLUSTER_SPREAD = 10              # Spread within clusters

GAMMA_QOS  = 0.5   # 0 = ignore reliability, 0.25–0.75 = mild, 1.0 = strong
LAMBDA_VOL = 0.25  # volatility penalty weight

# ---------- Pareto utilities on tuples ----------
def pareto_prune(objs):
    """Return only non-dominated points from a list of (err, cost)."""
    if not objs: return []
    idx = nondominated_indices(objs)
    return [objs[i] for i in idx]

def clip01_pair(p):  # safety against tiny overshoots
    return (min(max(p[0], 0.0), 1.0), min(max(p[1], 0.0), 1.0))

# ---------- Indicators ----------
def hypervolume_2d(front, ref=(1.0, 1.0)):
    if not front: return 0.0
    pts = pareto_prune([clip01_pair(p) for p in front])
    if not pts: return 0.0
    pts = sorted(pts, key=lambda a: a[0], reverse=True)  # sort by x desc
    hv, prev_x, floor_y = 0.0, ref[0], ref[1]
    for x, y in pts:
        hv += max(prev_x - x, 0.0) * max(ref[1] - floor_y, 0.0)
        prev_x = x
        floor_y = min(floor_y, y)  # maintain lower envelope
    # NOTE: no extra slice to x=0 for minimization
    return hv


def euclid(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)

def igd(front, ref_set):
    """
    Inverted Generational Distance (min): average distance from each ref point
    to its nearest point in `front`.
    """
    if not ref_set: return float('nan')
    if not front:   return float('inf')
    return float(np.mean([min(euclid(r, p) for p in front) for r in ref_set]))

def epsilon_additive(front, ref_set):
    """
    Additive epsilon indicator (min). Smallest ε such that
    ∀r∈ref_set, ∃p∈front with p_i ≤ r_i + ε for all i.
    """
    if not ref_set: return float('nan')
    if not front:   return float('inf')
    eps_per_ref = []
    for r in ref_set:
        # for this reference point, best we can do:
        eps_r = min(max(p[0]-r[0], p[1]-r[1]) for p in front)
        eps_per_ref.append(eps_r)
    return float(max(0.0, max(eps_per_ref)))


# front_log[alg][type_error][t] = list of (err, cost)
front_log = defaultdict(lambda: {'tp': defaultdict(list), 'res': defaultdict(list)})

def record_front(alg: str, type_error: str, t: int, objs: list[tuple[float,float]]):
    # store non-dominated only (keeps it compact)
    front_log[alg][type_error][t] = pareto_prune([clip01_pair(p) for p in objs])

def reference_set_for_time(type_error: str, t: int):
    union = []
    for alg in ['nsga','mopso','mogwo','greedy']:
        union.extend(front_log[alg][type_error].get(t, []))
    return pareto_prune(union)

def compute_indicators_for_run():
    # returns: metrics[alg][te] -> dict with arrays over time
    metrics = defaultdict(lambda: {'tp': {}, 'res': {}})
    for te in ['tp','res']:
        for t in range(num_times):
            ref = reference_set_for_time(te, t)
            for alg in ['nsga','mopso','mogwo','greedy']:
                front = front_log[alg][te].get(t, [])
                HV  = hypervolume_2d(front, ref=(1.0,1.0))
                IGD = igd(front, ref)
                EPS = epsilon_additive(front, ref)
                # append to arrays (create lazily)
                for name, val in [('HV',HV), ('IGD',IGD), ('EPS',EPS)]:
                    metrics[alg][te].setdefault(name, []).append(val)
    return metrics


# ---- RNG POOL -------------------------------------------------
MASTER_SEED = 42
_NUM_TIMES  = num_times  # keep in sync with your num_times

_root = np.random.SeedSequence(MASTER_SEED)

# spawn stable, independent streams for each “scope”
ss_global, ss_ou, ss_init, ss_greedy, ss_nsga, ss_mopso, ss_mogwo = _root.spawn(7)

RNG_GLOBAL = np.random.Generator(np.random.PCG64(ss_global))
RNG_OU     = [np.random.Generator(np.random.PCG64(s)) for s in ss_ou.spawn(_NUM_TIMES)]
RNG_INIT   = np.random.Generator(np.random.PCG64(ss_init))

RNG_GREEDY = [np.random.Generator(np.random.PCG64(s)) for s in ss_greedy.spawn(_NUM_TIMES)]
RNG_NSGA   = [np.random.Generator(np.random.PCG64(s)) for s in ss_nsga.spawn(_NUM_TIMES)]
RNG_MOPSO  = [np.random.Generator(np.random.PCG64(s)) for s in ss_mopso.spawn(_NUM_TIMES)]
RNG_MOGWO  = [np.random.Generator(np.random.PCG64(s)) for s in ss_mogwo.spawn(_NUM_TIMES)]

_OU_NODE_RNGS = {}

def rng_for(scope: str, t: int = None, idx: int = None):
    if scope == 'greedy':              return RNG_GREEDY[t]
    if scope == 'nsga':                return RNG_NSGA[t]
    if scope in ('mopso', 'pso'):      return RNG_MOPSO[t]
    if scope in ('mogwo', 'gwo'):      return RNG_MOGWO[t]
    if scope in ('init', 'ou_init'):   return RNG_INIT
    if scope == 'coords':
        # per-time deterministic stream for coordinate sampling (if you use it)
        return np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([MASTER_SEED, 2001, int(t)])
        ))
    if scope == 'ou_node':
        idx = int(idx)
        if idx not in _OU_NODE_RNGS:
            _OU_NODE_RNGS[idx] = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([MASTER_SEED, 1001, idx]))
            )
        return _OU_NODE_RNGS[idx]
    return RNG_GLOBAL

# ----------------------------------------------------------------


word_to_num = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10
}

curr_ratio = 'three_one'
left_str, right_str = curr_ratio.split('_')


def num_providers_consumers(type_provider):
    if type_provider == 'producer':
        return round(num_services * (word_to_num[left_str] / (word_to_num[left_str] + word_to_num[right_str])))  # e.g., suppose we have 'three_one' -> then this would return 75
    return round(num_services * (word_to_num[right_str] / (word_to_num[left_str] + word_to_num[right_str])))

coverage_fraction = 0.2
space_diag = math.sqrt(SPACE_SIZE[0]**2 + SPACE_SIZE[1]**2)
COVERAGE_RADIUS = space_diag * coverage_fraction

num_services   = 100
num_providers  = int(num_providers_consumers('producer'))   # these are defined accord to 'curr_ratio' above
print(num_providers)
num_consumers  = int(num_providers_consumers('consumer'))
print(num_consumers)

# Enhanced NSGA-II Hyperparameters



population_size      = 120    # Increased population size for better diversity
crossover_prob_range = (0.85, 0.95) # Slightly lower minimum to allow more exploitation
mutation_prob_range  = (0.04, 0.08) # Wider range for more exploration
crossover_eta        = 15      # Lower eta for more diverse crossover
mutation_eta         = 30     # Lower eta for more aggressive mutations
tournament_size      = 2       # Larger tournament for better selection pressure
max_generations      = 300
patience             = 75        # More patience for better convergence
adaptive_mutation    = True      # Enable adaptive mutation rate
#diversity_threshold  = 0.1       # Minimum diversity threshold
num_iterations = 1

cost_min, cost_max = float('inf') , 0.0

global_cost_min, global_cost_max = 0.0, 0.0
# Data Processing Here (Loading in Datasets):

import pandas as pd 
import os
print(os.getcwd())
pathway = "qws2.txt"

qos_states = ["Low", "Medium", "High"]


rs_two = []
tp_two = []

greedy_scs_res_cache = [0]
greedy_scs_tp_cache = [0]
nsga_scs_res_cache = [0]
nsga_scs_tp_cache = [0]
mopso_scs_res_cache = [0]
mopso_scs_tp_cache = [0]
mogwo_scs_res_cache = [0]
mogwo_scs_tp_cache = [0]
random_scs_res_cache = [0]
random_scs_tp_cache = [0]

greedy_scs_dict = {}
nsga_scs_dict = {}
mopso_scs_dict = {}
mogwo_scs_dict = {}
random_scs_dict = {}

with open(pathway, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        fields = line.strip().split(',')   # if fields are comma-separated
        # Make sure there are at least 3 elements in the line
        if len(fields) >= 3:
            rs_two.append(fields[0])
            tp_two.append(fields[2])

def processing(path):
    df = pd.read_csv(path, sep='\s+', header=None, engine='python')
    mat = df.to_numpy()
    return mat[:, :-1] if mat.shape[1] == 5826 else mat


try:
    response_times = processing("/Users/evanbarker/Desktop/dataset1/rtMatrix.txt")
    throughputs = processing("/Users/evanbarker/Desktop/dataset1/tpMatrix.txt")
except FileNotFoundError:
    # Synthetic fallback
    print("Using synthetic data since data files were not found")
    response_times = RNG_INIT.random((num_services, num_services))
    throughputs    = RNG_INIT.random((num_services, num_services))



df_rt = pd.DataFrame(response_times)
df_tp = pd.DataFrame(throughputs)
service_rt_list = (df_rt.median(axis=0) * 1000).tolist()
service_tp_list = df_tp.median(axis=0).tolist()

service_ids     = [f"S{i+1}" for i in range(num_services)]
providers       = service_ids[:num_providers]
consumers       = service_ids[num_providers:]
record_list_dict: dict = {}

largest_cost_all = [] # stores largest costs (max cost) for all 30 time frames

t_ref = []

# OU parameters
OU_THETA = 0.1
OU_SIGMA = 5.0
DELTA_T  = 1.0  # fixed for simplicity

# Core OU process update
def OU_step(x_t, mu, theta=OU_THETA, sigma=OU_SIGMA, delta_t=DELTA_T, rng=None):
    rng = rng or RNG_GLOBAL
    noise = rng.standard_normal(size=x_t.shape)
    return x_t + theta * (mu - x_t) * delta_t + sigma * np.sqrt(delta_t) * noise

# OU trajectory generator for one node
def generate_OU_trajectory(start_pos, mu, num_steps, rng=None):
    rng = rng or RNG_GLOBAL
    traj = [start_pos]
    for _ in range(num_steps):
        next_pos = OU_step(traj[-1], mu, rng=rng)
        next_pos = np.clip(next_pos,
                           [0.0, 0.0],
                           [float(SPACE_SIZE[0]), float(SPACE_SIZE[1])])
        traj.append(next_pos)
    return np.array(traj)

def init_node_positions_and_means(total_nodes, distribution='uniform', rng=None):
    rng = rng or RNG_GLOBAL
    start_positions, means = [], []

    if distribution == 'uniform':
        # independent ranges per axis
        for _ in range(total_nodes):
            pos = rng.uniform(low=[0.0, 0.0],
                              high=[float(SPACE_SIZE[0]), float(SPACE_SIZE[1])],
                              size=(2,))
            start_positions.append(pos)
            means.append(np.array(SPACE_SIZE, dtype=float) / 2.0)

    elif distribution == 'random':
        for _ in range(total_nodes):
            pos = rng.random(2) * np.array(SPACE_SIZE, dtype=float)
            start_positions.append(pos)
            means.append(np.array(SPACE_SIZE, dtype=float) / 2.0)

    elif distribution == 'clumped':
        cluster_centers = rng.random((NUM_CLUSTERS, 2)) * np.array(SPACE_SIZE, dtype=float)
        for _ in range(total_nodes):
            center = cluster_centers[rng.integers(0, NUM_CLUSTERS)]
            pos = rng.normal(loc=center, scale=CLUSTER_SPREAD, size=2)
            pos = np.clip(pos,
                          [0.0, 0.0],
                          [float(SPACE_SIZE[0]), float(SPACE_SIZE[1])])
            start_positions.append(pos)
            means.append(center)  # cluster center as OU mean
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return np.array(start_positions), np.array(means)

def generate_all_OU_trajectories(distribution):
    total_nodes = num_providers + num_consumers

    # deterministic RNG for initial placement
    rng_init = rng_for('ou_init')
    starts, mus = init_node_positions_and_means(total_nodes, distribution, rng=rng_init)

    traj_dict = {}
    for i in range(total_nodes):
        # deterministic, independent stream per node
        rng_node = rng_for('ou_node', idx=i)
        traj = generate_OU_trajectory(starts[i], mus[i], num_steps=num_times, rng=rng_node)
        sid = f"p{i}" if i < num_providers else f"c{i - num_providers}"
        traj_dict[sid] = traj
    return traj_dict



def euclidean_distance(p, c):   # outputs the l2 (euclidean) norm
    px, py = p['coords']
    cx, cy = c['coords']
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

def generate_coordinates(distribution, total_nodes, time_index, rng=None):
    # If you don't pass one in, derive a deterministic stream per time slice
    if rng is None:
        rng = rng_for('coords', time_index)

    if distribution == 'uniform':
        # independent uniform per axis
        low  = np.array([0.0, 0.0])
        high = np.array([float(SPACE_SIZE[0]), float(SPACE_SIZE[1])])
        coords = rng.uniform(low, high, size=(total_nodes, 2))

    elif distribution == 'random':
        # in [0,1) then scale by box size (works for non-square too)
        coords = rng.random((total_nodes, 2)) * np.array(SPACE_SIZE, dtype=float)

    elif distribution == 'clumped':
        cluster_centers = rng.random((NUM_CLUSTERS, 2)) * np.array(SPACE_SIZE, dtype=float)
        coords = []
        for _ in range(total_nodes):
            center = cluster_centers[rng.integers(0, NUM_CLUSTERS)]
            point  = rng.normal(loc=center, scale=CLUSTER_SPREAD, size=2)
            point  = np.clip(point, [0.0, 0.0], [SPACE_SIZE[0], SPACE_SIZE[1]])
            coords.append(point)
        coords = np.array(coords, dtype=float)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return coords



def collect_raw_qos_sequences(rt_list, tp_list, num_times, num_services_total, num_providers, states=("Low","Medium","High")):
    provider_qos_raw = {i: ["Medium"] for i in range(num_providers)}  # seed


    max_rt = max(float(x) for x in rt_list) if rt_list else 1.0
    max_tp = max(float(x) for x in tp_list) if tp_list else 1.0


    for t in range(num_times):
        

        rng = rng_for('init', t)  # or rng_for('init', t) if you made a per-time stream

        k_rt = min(len(rt_list), num_services_total)
        k_tp = min(len(tp_list), num_services_total)

        # fastest + clean: permutation then slice
        rt_indices = rng.permutation(len(rt_list))[:k_rt]
        tp_indices = rng.permutation(len(tp_list))[:k_tp]


        for i in range(num_services_total):
            ri = rt_indices[i % len(rt_indices)]
            rj = tp_indices[i % len(tp_indices)]
            resp = float(rt_list[ri])
            thrp = float(tp_list[rj])

            if i < num_providers:
                resp_norm = resp / max_rt if max_rt > 0 else 0.5
                thrp_norm = thrp / max_tp if max_tp > 0 else 0.5  # your convention
                inferred = classify_qos(resp_norm, thrp_norm)
                provider_qos_raw[i].append(inferred)

    return provider_qos_raw


def construct_transition_matrix(provider_qos_raw, states=("Low","Medium","High"), laplace=1.0):
    counts = {s: defaultdict(int) for s in states}
    for seq in provider_qos_raw.values():
        for t in range(1, len(seq)):
            prev, curr = seq[t-1], seq[t]
            counts[prev][curr] += 1
    T = {}
    for prev in states:
        total = sum(counts[prev][s] + laplace for s in states)
        T[prev] = {s: (counts[prev][s] + laplace) / total for s in states}
    return T



def classify_qos(resp_norm, thrp_norm, alpha=0.5):
    score = alpha * resp_norm + (1-alpha) * (1-thrp_norm)
    if score >= 0.66: return "Low"
    elif score >= 0.33: return "Medium"
    else:return "High"

    
# Temporal smoothing via Markov transition model


def smooth_qos(prev_qos, inferred_qos, transition_matrix, beta, rng=None):
    rng = rng or RNG_GLOBAL
    prob_next = transition_matrix[prev_qos]              # dict
    adjusted = prob_next.copy()
    adjusted[inferred_qos] = adjusted.get(inferred_qos, 0) + beta
    total = sum(adjusted.values())
    states = list(adjusted.keys())
    probs  = np.array([adjusted[s] for s in states], dtype=float) / total
    return rng.choice(states, p=probs)

class Individual:
    def __init__(self, genes: List[int]):
        self.genes = genes[:]  # provider indices
        self.error = None
        self.cost  = None
        self.rank = float('inf')
        self.crowding = 0.0

    def evaluate(self, prods, cons, type_error, max_err,t):
        algorithm = "nsga"
        curr_max_cost = max(cost_per_dict[f"{t}"])    # use this for per-time normalization!
        curr_min_cost = min(cost_per_dict[f"{t}"])
        errs = []
        costs = []
        for idx_c, c in enumerate(cons):
            p = prods[self.genes[idx_c]]
            raw  = (QoS_tp_error if type_error=='tp' else QoS_res_error if type_error=='res' else relative_error)(p, c, algorithm=algorithm)
            errs.append(norm_err(type_error, raw,t))
            r = p.get('qos_prob', 0.5)
            v = p.get('qos_volatility', 0.0)
            costs.append(((p['cost'] - curr_min_cost) / (curr_max_cost - curr_min_cost + 1e-12))
                        * (1.0 + LAMBDA_VOL * v) / (max(r,1e-6)**GAMMA_QOS))
        self.error = float(np.mean(errs))
        self.cost  = float(np.mean(costs))

def assign_rank_and_crowding(pop):
    """
    Runs fast_non_dominated_sort and calculate_crowding_distance,
    then writes .rank and .crowding into each Individual.
    """
    fronts = fast_non_dominated_sort(pop)
    for rank, front in enumerate(fronts):
        cd_pairs = calculate_crowding_distance(front)
        for ind, cd in cd_pairs:
            ind.rank     = rank
            ind.crowding = cd
    return fronts


def initiate_records(rt_list, tp_list, learned_T, num_times, num_services_total, num_providers):
    global global_cost_min, global_cost_max, cost_per_dict

    alpha, beta = 0.5, 0.5
    raw_costs = []
    record_list_dict.clear()

    max_rt = max(float(i) for i in rt_list)
    max_tp = max(float(i) for i in tp_list)

    d1 = {}
    num_services_total = len(service_ids)

    # make OU deterministic too (see section 4 for RNG-enabled OU helpers)
    all_trajectories = generate_all_OU_trajectories(SPATIAL_DISTRIBUTION)

    provider_qos_smoothed = {i: ["Medium"] for i in range(num_providers)}

    for t in range(num_times):
        rng = rng_for('init', t)

        # draw fresh permutations each time step
        k_rt = min(len(rt_list), num_services_total)
        k_tp = min(len(tp_list), num_services_total)
        rt_indices = rng.permutation(len(rt_list))[:k_rt]
        tp_indices = rng.permutation(len(tp_list))[:k_tp]

        prods, cons = [], []
        cost_per = []

        coords = []
        for i in range(num_services_total):
            sid = f"p{i}" if i < num_providers else f"c{i - num_providers}"
            coords.append(all_trajectories[sid][t])  # coordinate at time t

        for i, sid in enumerate(service_ids):
            ri = rt_indices[i % len(rt_indices)]
            rj = tp_indices[i % len(tp_indices)]
            resp = float(rt_list[ri])
            thrp = float(tp_list[rj])

            # defaults (used for consumers)
            qos_prob = 0.5
            qos_vol  = 0.0
            QoS      = None
            cost     = 0.0

            if i < num_providers:
                # example synthetic cost; use rng
                cost = float(rng.random())
                raw_costs.append(cost)
                cost_per.append(cost)

                p_index    = i
                resp_norm  = resp / max_rt if max_rt > 0 else 0.5
                thrp_norm  = 1.0 - (thrp / max_tp) if max_tp > 0 else 0.5
                inferred_qos = classify_qos(resp_norm, thrp_norm)

                prev_qos      = provider_qos_smoothed[p_index][-1] if t > 0 else "Medium"
                smoothed_qos  = smooth_qos(prev_qos, inferred_qos, learned_T, beta=0.3)
                provider_qos_smoothed[p_index].append(smoothed_qos)

                QoS = smoothed_qos
                probs_next = learned_T[prev_qos]
                qos_prob   = probs_next.get("Medium", 0.0) + probs_next.get("High", 0.0)
                qos_vol    = 1.0 - sum(p*p for p in probs_next.values())

            rec = {
                "service_id": sid,
                "timestamp": t,
                "response_time_ms": resp,
                "throughput_kbps":  thrp,
                "cost":             cost,
                "coords":           coords[i].tolist(),
                "qos":              QoS,
                "qos_prob":         qos_prob,
                "qos_volatility":   qos_vol
            }
            (prods if i < num_providers else cons).append(rec)

        d1[f"{t}"] = cost_per
        record_list_dict[t] = [prods, cons]

    cost_per_dict = d1

    


# --- QoS & ERRORS -------------------------------------------
    
keepTrack_nsga_tp = {}
keepTrack_mopso_tp = {}
keepTrack_mogwo_tp = {}
keepTrack_nsga_res = {}
keepTrack_mopso_res = {}
keepTrack_mogwo_res = {}



#consumer_ids = [f"c{i+71}" for i in range(num_consumers)]

consumer_ids = [f"c{i+1}" for i in range(num_consumers)]   # this is rock solid

def producer_exists(streak_dict, c_id, t, p_id):
    return any(p == p_id for p, _ in streak_dict.get(c_id, {}).get(t, []))

def init_streaks(consumer_ids, num_times):
    return {
        c_id: {t: [] for t in range(num_times)}
        for c_id in consumer_ids
    }

def init_marker(consumer_ids):
    return {
        c_id : [] for c_id in consumer_ids
    }

def SID_to_c_s_id(sid):  # This will give S1, S2, ..., S100.
    if int(sid[1:]) <= num_providers:
        return str("p" + str(int(sid[1:])))
    return str("c"+str(int(sid[1:])-num_providers))
    # if int(sid[1:]) <= 70:
    #     return str("p" + str(int(sid[1:])))
    # return str("c" + str(int(sid[1:])))

# Define keys as (algorithm, metric)
keys = [
    ("greedy", "res"),
    ("greedy", "tp"),
    ("nsga",   "res"),
    ("nsga",   "tp"),
    ("mopso",  "res"),
    ("mopso",  "tp"),
    ("mogwo",  "res"),
    ("mogwo",  "tp"),
]

# Centralized dictionary of all streaks
all_streaks = {
    f"{alg}_{metric}_streaks": init_streaks(consumer_ids, num_times)
    for alg, metric in keys
}

all_streaks_markers = {
    f"{alg}_{metric}_streaks_markers": init_marker(consumer_ids)
    for alg, metric in keys
}

def get_streaks_at_time(all_streaks, alg_metric_key: str, time_step: int):  # fetches a dict full of per-time consumer rolling tuple records
    time_slice = {}
    streak_dict = all_streaks[alg_metric_key]
    for consumer_id, time_dict in streak_dict.items():
        if time_step in time_dict:
            time_slice[consumer_id] = time_dict[time_step]
        else:
            time_slice[consumer_id] = []
    return time_slice

def get_tuple_counts_and_avg_at_time(streaks, time_step, time_dict):
    consumer_counts = {
        consumer_id: len(time_dict.get(time_step, []))
        for consumer_id, time_dict in streaks.items()
    }
    total = sum(consumer_counts.values())
    avg = total / len(consumer_counts) if consumer_counts else 0
    return consumer_counts, avg



def update_streaks_with_marker(
    alg: str,
    metric: str,
    t: int,
    c_id: str,
    p_id: str,
    all_streaks=all_streaks,
    all_streaks_markers=all_streaks_markers
):
    key = f"{alg}_{metric}_streaks"
    marker_key = f"{alg}_{metric}_streaks_markers"
   
    streak_dict = all_streaks[key]
    time_streaks = streak_dict[c_id]

    # Step 1: Propagate history
    if t > 0:
        time_streaks[t] = time_streaks[t - 1].copy()   # carry over/ roll over previous time frame tuples.
    else:
        time_streaks[t] = []

    # Step 2: Check if producer was already present in t-1
    existing_producers = {p for p, _ in time_streaks[t - 1]} if t > 0 else set()
    producer_was_present = p_id in existing_producers

    if producer_was_present:
        # below code will not throw error right away because 'if all_streaks_markers[marker_key][c_id][-1]' was placed at the front -> means condition will return a falsy and thus block right after will run.
        if all_streaks_markers[marker_key][c_id][-1] and p_id == all_streaks_markers[marker_key][c_id][-1]:  # this was the same producer for the previous time frame... so, we append to the END of the tuple set for 'all_streaks' -> ensures we are distinguishing tuples housing same producers correctly
            for i in reversed(range(len(time_streaks[t]))):
                producer, count = time_streaks[t][i]
                if producer == p_id:
                    time_streaks[t][i] = (producer, count + 1)
                    all_streaks_markers[marker_key][c_id].append(p_id)
                    break
        elif p_id != all_streaks_markers[marker_key][c_id][-1]:  # here, we are appending (p_id,1) even though p_id is indeed inside of time_streaks for the reason that 'p_id != all_streaks_markers[marker_key][-1]' held true.
            time_streaks[t].append((p_id,1))
            all_streaks_markers[marker_key][c_id].append(p_id)
    else:
        time_streaks[t].append((p_id,1))
        all_streaks_markers[marker_key][c_id].append(p_id)



def add_to_cache(algorithm, cache_index, type, p=None, c=None):
    hit = 1
    if p != None and c != None:    # restricted to only nsga/mopso/mogwo
        c_id = c['service_id']
        if cache_index not in globals()[f"keepTrack_{algorithm}_{type}"]:
            globals()[f"keepTrack_{algorithm}_{type}"][cache_index] = [c_id]
        if c_id not in globals()[f"keepTrack_{algorithm}_{type}"][cache_index]:
            globals()[f"keepTrack_{algorithm}_{type}"][cache_index].append(c_id)
        elif c_id in globals()[f"keepTrack_{algorithm}_{type}"][cache_index]:
            return                                                      # i.e., DONT HIT THE CACHE if c in dict already! This means that this consumer already has coverage and we've already accounted for that!
    dict_fetch = globals()[f"{algorithm}_scs_{type}_cache"]
    while len(dict_fetch) <= cache_index:
        dict_fetch.append(0)                # ensures we don't get error for line below
    dict_fetch[cache_index] += hit
        
def reg_err(p, c, type):
    if type == "tp":
        prov, req = p["throughput_kbps"], c["throughput_kbps"]
        if prov < req:
            return (req - prov) / (req + 1e-12)
        elif req < prov:
            log_input = 0.005 * ((prov - req) / (req + 1e-12))
            safe_log_input = max(log_input, 1e-6)  # strictly positive
            to_add = abs(0.5 * math.log(safe_log_input))
            return to_add
        else:
            return 0
    elif type == "res":
        prov, req = p["response_time_ms"], c["response_time_ms"]
        if prov < req:
            log_input = 0.005 * ((req - prov) / (req + 1e-12))
            safe_log_input = max(log_input, 1e-6)  # strictly positive
            to_add = abs(0.5 * math.log(safe_log_input))
            return to_add
        elif req < prov:
            return (prov-req) / (req + 1e-12)
        else: 
            return 0
    
def QoS_tp_error(p, c, algorithm='', time=0, final=False, marker = True):    

    if final == False:
        return reg_err(p, c, type="tp")

    prov, req = p["throughput_kbps"], c["throughput_kbps"]

    new_cache = False  # default assumption

    if marker == True:
        cache_index = int(time)
        marker = False

    if prov >= req:
        type = "tp"
        if algorithm == 'nsga' or algorithm == 'mopso' or algorithm == 'mogwo':
            add_to_cache(algorithm, cache_index, type, p=p, c=c)
        else:
            add_to_cache(algorithm, cache_index, type)
        
    if algorithm == "greedy":
        return reg_err(p,c,type="tp")


def QoS_res_error(p,c, algorithm='', time=0, final=False, marker=True):

    if final == False:
        return reg_err(p, c, type="res")
    
    new_cache = False  # default assumption
   
    if marker == True:
        cache_index = int(time)
        marker = False
    
    prov, req = p["response_time_ms"], c["response_time_ms"]
    
    if req >= prov:
        type = "res"
        if algorithm == 'nsga' or algorithm == 'mopso' or algorithm == 'mogwo':
            add_to_cache(algorithm, cache_index, type, p=p, c=c)
        else:
            add_to_cache(algorithm, cache_index, type)

    if algorithm == "greedy":
        return reg_err(p,c,type="res")
    

def relative_error(p, c): return abs(1 - ((p["response_time_ms"] + p["throughput_kbps"])/(c["response_time_ms"] + c["throughput_kbps"]))) if (c["response_time_ms"] + c["throughput_kbps"]) != 0 else 0.0
    
# synethic cost function which is just an equal weighted avg of cost of producer and consumer nodes when they interact.
def synthetic_cost_function(p,c, curr_time): return p["cost"]

# def norm_err(type, error, time):
#     if type == 'tp':
#         return ( error - max_min_obj[f"{time}"][2] ) / ( max_min_obj[f"{time}"][0] - max_min_obj[f"{time}"][2] ) 
#     else:
#         return (error - max_min_obj[f"{time}"][3]) / ( max_min_obj[f"{time}"][1] - max_min_obj[f"{time}"][3] ) 
    
def norm_err(kind, error, time):
    mx_tp, mx_res, mn_tp, mn_res = max_min_obj[f"{time}"]
    if kind == 'tp':
        denom = (mx_tp - mn_tp) or 1e-12
        return (error - mn_tp) / denom
    else:
        denom = (mx_res - mn_res) or 1e-12
        return (error - mn_res) / denom

        


def greedy_algorithm(type_error: str, record_cb=None) -> Tuple[List[float], List[float], List[float]]:
    errors, costs, error_stds = [], [], []
    alpha = 0.5
    err_func_map = {
        'tp': QoS_tp_error,
        'res': QoS_res_error,
        'rel': relative_error
    }
    err_func = err_func_map.get(type_error, relative_error)

    for t in range(num_times):

        curr_max_cost = max(cost_per_dict[f"{t}"])    # use this for per-time normalization!
        curr_min_cost = min(cost_per_dict[f"{t}"])

        rng = rng_for('greedy', t)
        prods, cons = record_list_dict[t]
        total_err, total_cost = 0.0, 0.0
        matched_pairs = []

        for c in cons:
            best_score = float('inf')
            best_p = None
            scores = []
            valid_prov_indices = []
            for i,p in enumerate(prods):
                # Skip if producer and consumer are the same
                if euclidean_distance(p,c) > COVERAGE_RADIUS or p == c:   # if consumer outside the producer coverage range, not a viable option.
                    continue
                
                r = p.get('qos_prob', 0.5)      # reliability prior
                v = p.get('qos_volatility', 0)  # volatility penalty

                score = (
                    norm_err(type_error, err_func(p, c, time=t), t)
                    * ((p['cost'] - curr_min_cost) / (curr_max_cost - curr_min_cost + 1e-12))
                    * (1.0 + LAMBDA_VOL * v)
                    / (max(r, 1e-6) ** GAMMA_QOS)
                )
               
                scores.append(score)
                valid_prov_indices.append(i)

            if not valid_prov_indices:
                raise ValueError(f"No producers within coverage for consumer {c['service_id']} at time {t}")
            min_idx = valid_prov_indices[np.argmin(scores)]
            best_p = prods[min_idx]     

            c_id = SID_to_c_s_id(c["service_id"])
            p_id = SID_to_c_s_id(best_p["service_id"])

            update_streaks_with_marker('greedy', type_error, t, c_id, p_id)
            #print(best_p is None)
            #indices = list(range(len(prods)))
            #print(indices)
            #random.shuffle(indices)
            #min_idx = min(indices, key=lambda i: scores[i])
            #best_p = prods[min_idx]
            err_val = err_func(best_p, c, algorithm="greedy", time = t, final=True)
            err_norm = norm_err(type_error, err_val,t)
            total_err += err_norm
            total_cost += ((best_p['cost'] - curr_min_cost)/(curr_max_cost-curr_min_cost))
            matched_pairs.append((err_norm, best_p['cost']))

            # if best_p is not None:
            #     err_val = err_func(best_p, c)
            #     err_norm = norm_err(type_error, err_val)
            #     total_err += err_norm
            #     total_cost += ((best_p['cost'] - curr_min_cost)/(curr_max_cost-curr_min_cost))
            #     cost_normed = ((best_p['cost'] - curr_min_cost)/(curr_max_cost-curr_min_cost))
            #     matched_pairs.append((err_norm, cost_normed))

        avg_err = total_err / len(cons)
        avg_cost = total_cost / len(cons)
        if record_cb:
            record_cb('greedy', type_error, t, [(avg_err, avg_cost)])
        err_std = np.std([e for e, _ in matched_pairs]) if len(matched_pairs) > 1 else 0.0

        errors.append(avg_err)
        costs.append(avg_cost)
        error_stds.append(err_std)

    return errors, costs, error_stds


# Random Algorithm:

# def random_algorithm(type_error: str) -> Tuple[List[float], List[float], List[float]]:
#     errors = []
#     costs = []
#     error_stds = []
#     for t in range(num_times):
#         prods, cons = record_list_dict[t]
#         matched_pairs = []
#         for c in cons:
#             found_one = False
#             p = random.choice(prods)
#             while not found_one:    # Ensuring that we don't have the same node for producers/consumers in network.
#                 if c != p:
#                     found_one = True
#                     break
#                 p = random.choice(prods)
#             err = (QoS_tp_error if type_error=='tp' else QoS_res_error if type_error=='res' else relative_error)(p, c, "random")
#             if type_error == 'tp':
#                 err = (err - tp_min) / (tp_max - tp_min + 1e-12)   # :contentReference[oaicite:0]{index=0}
#             else:
#                 err = (err - res_min) / (res_max - res_min + 1e-12) # :contentReference[oaicite:1]{index=1}
#             matched_pairs.append((err, p['cost']))
        
#         err_values = [pair[0] for pair in matched_pairs]
#         cost_values = [pair[1] for pair in matched_pairs]
        
#         avg_err = np.mean(err_values)
#         avg_cost = np.mean(cost_values)
#         err_std = np.std(err_values) if len(err_values) > 1 else 0.0
        
#         errors.append(avg_err)
#         costs.append(avg_cost)
#         error_stds.append(err_std)
        
#     return errors, costs, error_stds

# Helper Functions below

def get_hashable_ind(ind):
    """Convert individual to hashable form for comparison"""
    return (ind[0], ind[1]['service_id'], ind[2]['service_id'], ind[3])

def calculate_population_diversity(population):
    """Calculate diversity of population based on unique service pairings"""
    unique_pairs = set()
    for ind in population:
        unique_pairs.add((ind[1]['service_id'], ind[2]['service_id']))
    return len(unique_pairs) / len(population)


def enhanced_NSGA_II_for_time(t: int, type_error: str, weight: float) -> Tuple[List[Any], float, float, List[float], float]:


    curr_max_cost = max(cost_per_dict[f"{t}"])    # use this for per-time normalization!
    curr_min_cost = min(cost_per_dict[f"{t}"])
    rng = rng_for('nsga', t)
    
    prods, cons = record_list_dict[t]
    alpha = 0.5
    coverage_ok = all(
        any(euclidean_distance(p, c) <= COVERAGE_RADIUS for p in prods)
        for c in cons)
    if not coverage_ok:
        raise RuntimeError("Some consumers are unreachable given COVERAGE_RADIUS")

    err_func_map = {
        'tp': QoS_tp_error,
        'res': QoS_res_error,
        'rel': relative_error
    }
    err_func = err_func_map.get(type_error, relative_error)
    
    # decide which max error to normalize by
    max_err = max_tp_err if type_error=='tp' else max_res_err

    # instead of building pop as 4-tuples, do:
    pop: List[Individual] = []

    # Greedy seeding for first weight-fraction
    num_greedy = math.floor(weight * len(cons))

    # population startup here ensures we don't allow producer == consumer...
    while len(pop) < population_size:
        genes = []
        # for i, c in enumerate(cons):
        #     eligible_indices = [j for j, p in enumerate(prods) if p['service_id'] != c['service_id']]
        #     if not eligible_indices:
        #         # fallback: just use a random index (better than crash)
        #         selected_index = random.randint(0, len(prods) - 1)
        #     else:
        #         selected_index = random.choice(eligible_indices)
        #     genes.append(selected_index)
        # indiv = Individual(genes)
        # indiv.evaluate(prods, cons, type_error, max_err, t)
        # pop.append(indiv)
        j = 0
        for i, c in enumerate(cons):
            valid_provs = [j for j, p in enumerate(prods) if euclidean_distance(p,c) <= COVERAGE_RADIUS]
            if not valid_provs:
                raise ValueError(f"No valid providers for consumer {c['service_id']}")
            selected = int(rng.choice(valid_provs))
            genes.append(selected)


        
        indiv = Individual(genes)
        indiv.evaluate(prods, cons, type_error, max_err, t)
        pop.append(indiv)

                


    # Initial metrics for convergence tracking
    best_front = pareto_front(pop)
    
    stagnant = 0
    best_score = float('inf')
    for gen in range(1, max_generations+1):
        # pm = random.uniform(*mutation_prob_range)
        # # Crossover probability - slightly adaptive
        # pc = random.uniform(*crossover_prob_range)
        pm = rng.uniform(mutation_prob_range[0], mutation_prob_range[1])
        pc = rng.uniform(crossover_prob_range[0], crossover_prob_range[1])

        # Selection & variation
        offspring = []
        while len(offspring) < population_size:
            p1 = tournament_select(pop, tournament_size, rng)         # pick best producers from pop based on tournament_size 
            p2 = tournament_select(pop, tournament_size, rng)
            c1, c2 = sbx_crossover(p1, p2, crossover_eta, pc, curr_time=t, rng=rng)
            c1.evaluate(prods, cons, type_error, max_err,t)
            c2.evaluate(prods, cons, type_error, max_err,t)
            m1 = polynomial_mutation(c1, mutation_eta, pm, curr_time=t, rng=rng)
            m1.evaluate(prods, cons, type_error, max_err,t)
            offspring.append(m1)
            if len(offspring) < population_size:
                m2 = polynomial_mutation(c2, mutation_eta, pm,curr_time=t, rng=rng)
                m2.evaluate(prods, cons, type_error, max_err,t)
                offspring.append(m2)

        # Combine current pop and offspring
        combined = pop + offspring
        # 1) Fast non-dominated sort
        #fronts = fast_non_dominated_sort(combined)

        fronts = assign_rank_and_crowding(combined)

        # 2) Assign rank and crowding to every individual
        for rank, front in enumerate(fronts):
            # crowding returns List[(Individual, distance)]
            cd_pairs = calculate_crowding_distance(front)
            for ind, cd in cd_pairs:
                ind.rank     = rank
                ind.crowding = cd

        # 3) Now select the next generation of size population_size 
        #    by filling front by front (standard NSGA-II replacement)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= population_size:
                new_pop.extend(front)
            else:
                # take only as many as needed, sorted by descending crowding
                sorted_front = sorted(front, key=lambda ind: ind.crowding, reverse=True)
                remaining = population_size - len(new_pop)
                new_pop.extend(sorted_front[:remaining])
                break

        pop = new_pop

        current_front = pareto_front(pop)
        current_err   = np.mean([ind.error for ind in current_front])
        current_cost  = np.mean([ind.cost  for ind in current_front])
        
        # Check for improvement
        current_score =  math.sqrt(current_err**2 + current_cost**2)  # We primarily care about minimizing error
        
        if current_score < best_score:
            best_score = current_score
            best_front = current_front
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                print(f"Early stopping at generation {gen} due to no improvement")
                break
    

    curr_best_indv = best_front[0]   # ptr. to first ind obj in best_front

    for iter in best_front:  # iter is type 'Individual'
        if (iter.error <= curr_best_indv.error and iter.cost < curr_best_indv.cost) or (iter.error < curr_best_indv.error and iter.cost <= curr_best_indv.cost):
            curr_best_indv = iter

    for gene_base in range(len(curr_best_indv.genes)):
        c_ind = gene_base
        p_ind = curr_best_indv.genes[c_ind]
        act_cons = record_list_dict[t][1][c_ind]
        act_prod = record_list_dict[t][0][p_ind]
        QoS_tp_error(p=act_prod, c=act_cons, algorithm='nsga', time=t,final=True)
        QoS_res_error(p=act_prod, c=act_cons, algorithm='nsga', time=t,final=True)

         # NEW STREAK UPDATE
        update_streaks_with_marker('nsga', type_error, t,
            SID_to_c_s_id(act_cons["service_id"]),
            SID_to_c_s_id(act_prod["service_id"]))

    record_front('nsga', type_error, t, [(ind.error, ind.cost) for ind in best_front])

    # Compute mean error and cost on best front
    mean_err = np.mean([ind.error for ind in best_front])
    mean_cost = np.mean([ind.cost for ind in best_front])
    
    # Calculate standard deviation of error among solutions in best front
    std_err = np.std([ind.error for ind in best_front]) if len(best_front) > 1 else 0.0
    
    return best_front, mean_err, mean_cost, std_err

def fast_non_dominated_sort(pop: List[Any]) -> List[List[Any]]:
    N = len(pop)
    S = {i: [] for i in range(N)}      # domination sets
    n = {i: 0 for i in range(N)}       # domination count
    fronts_idx: List[List[int]] = [[]]
    
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            better_p = ((pop[p].error <= pop[q].error and pop[p].cost <= pop[q].cost)
                        and (pop[p].error < pop[q].error or pop[p].cost < pop[q].cost))
            better_q = ((pop[q].error <= pop[p].error and pop[q].cost <= pop[p].cost)
                        and (pop[q].error < pop[p].error or pop[q].cost < pop[p].cost))
            if better_p:
                S[p].append(q)
            elif better_q:
                n[p] += 1
        if n[p] == 0:
            fronts_idx[0].append(p)
    
    i = 0
    while fronts_idx[i]:
        next_front: List[int] = []
        for p_idx in fronts_idx[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    next_front.append(q_idx)
        i += 1
        fronts_idx.append(next_front)
    
    # Drop last empty front
    if not fronts_idx[-1]:
        fronts_idx.pop()

    # Convert indices to Individual lists
    return [[pop[idx] for idx in front] for front in fronts_idx]

# Crowding distance for a front of Individual objects

def calculate_crowding_distance(front: List[Any]) -> List[Tuple[Any, float]]:
    n = len(front)
    if n == 0:
        return []
    if n == 1:
        return [(front[0], float('inf'))]

    # Extract objective arrays
    errs = np.array([ind.error for ind in front], dtype=float)
    costs = np.array([ind.cost for ind in front], dtype=float)

    # Sort indices
    err_idx = np.argsort(errs)
    cost_idx = np.argsort(costs)

    dist = np.zeros(n, dtype=float)
    dist[err_idx[[0, -1]]] = np.inf
    dist[cost_idx[[0, -1]]] = np.inf

    if n > 2:
        # interior points
        err_range = errs[err_idx[-1]] - errs[err_idx[0]] + 1e-12
        cost_range = costs[cost_idx[-1]] - costs[cost_idx[0]] + 1e-12

        # error-axis contributions
        dist[err_idx[1:-1]] += (errs[err_idx[2:]] - errs[err_idx[:-2]]) / err_range
        # cost-axis contributions
        dist[cost_idx[1:-1]] += (costs[cost_idx[2:]] - costs[cost_idx[:-2]]) / cost_range

    # Pair individuals with distances and sort descending
    paired = list(zip(front, dist))
    paired.sort(key=lambda x: (math.isinf(x[1]), x[1]), reverse=True)  # returns highest distance pairs to lowest distance.
    return paired

# Pareto front helper

def pareto_front(pop: List[Any]) -> List[Any]:
    fronts = fast_non_dominated_sort(pop)
    return fronts[0] if fronts else []

# Tournament selection using rank & crowding

# def tournament_select(pop: List[Any], k: int) -> Any:
#     # pick k random
#     contenders = random.sample(pop, k)
#     # first by front rank, then by crowding distance
#     # Assume each individual has a .rank and .crowding set externally
#     # You can attach rank and crowding before selection
#     contenders.sort(key=lambda ind: (ind.rank, -ind.crowding))
#     return contenders[0]

def tournament_select(pop: List[Any], k: int, rng) -> Any:
    idx = rng.choice(len(pop), size=k, replace=False)
    contenders = [pop[int(i)] for i in idx]
    contenders.sort(key=lambda ind: (ind.rank, -ind.crowding))
    return contenders[0]


# --- Helper: Pareto dominance ---
def dominates(obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
    """Return True if obj1 dominates obj2 (minimize both)."""
    return (obj1[0] <= obj2[0] and obj1[1] <= obj2[1]) and (obj1[0] < obj2[0] or obj1[1] < obj2[1])

# --- Helper: filter non-dominated ---
def nondominated_indices(objs: List[Tuple[float, float]]) -> List[int]:
    """Return list of indices whose objs are non-dominated."""
    n = len(objs)
    is_nd = [True] * n
    for i in range(n):
        for j in range(n):
            if i != j and dominates(objs[j], objs[i]):
                is_nd[i] = False
                break
    return [i for i, flag in enumerate(is_nd) if flag]


# --- Helper: vectorized crowding distance for objective tuples ---
def crowding_distance(objs: List[Tuple[float, float]]) -> List[float]:
    """Compute crowding distance for a list of (err, cost), return distances in same order."""
    n = len(objs)
    if n == 0:
        return []
    if n == 1:
        return [float('inf')]

    errs = np.array([o[0] for o in objs], dtype=float)
    costs = np.array([o[1] for o in objs], dtype=float)
    dist = np.zeros(n, dtype=float)

    # sort indices
    err_idx = np.argsort(errs)
    cost_idx = np.argsort(costs)
    # boundaries infinite
    dist[err_idx[[0, -1]]] = np.inf
    dist[cost_idx[[0, -1]]] = np.inf

    if n > 2:
        # error axis
        err_range = errs[err_idx[-1]] - errs[err_idx[0]] + 1e-12
        delta_err = (errs[err_idx[2:]] - errs[err_idx[:-2]]) / err_range
        dist[err_idx[1:-1]] += delta_err
        # cost axis
        cost_range = costs[cost_idx[-1]] - costs[cost_idx[0]] + 1e-12
        delta_cost = (costs[cost_idx[2:]] - costs[cost_idx[:-2]]) / cost_range
        dist[cost_idx[1:-1]] += delta_cost

    return dist.tolist()

def compute_diversity(pop: np.ndarray) -> float:
    # pop: (swarm_size, num_cons) array of provider indices
    total = 0
    count = 0
    for a, b in combinations(pop, 2):
        total += np.sum(a != b)
        count += len(a)
    return (total / count) / (len(pop)*(len(pop)-1)/2)

def sbx_crossover(parent1: Individual,
                  parent2: Individual,
                  eta: float,
                  pc: float,
                  curr_time,
                  rng,
                  max_attempts: int = 10) -> Tuple[Individual, Individual]:
    n = len(parent1.genes)
    child1, child2 = [0] * n, [0] * n
    provs, cons = record_list_dict[curr_time]

    if rng.random() > pc:
        child1 = [g if g != i else (g + 1) % num_providers for i, g in enumerate(parent1.genes)]
        child2 = [g if g != i else (g + 1) % num_providers for i, g in enumerate(parent2.genes)]
        return Individual(child1), Individual(child2)

    for i in range(n):
        g1, g2 = parent1.genes[i], parent2.genes[i]
        if rng.random() <= 0.5 and abs(g1 - g2) > 1e-6:
            attempt = 0
            while attempt < max_attempts:
                u = rng.random()
                beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 else (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                v1 = 0.5 * ((1 + beta) * g1 + (1 - beta) * g2)
                v2 = 0.5 * ((1 - beta) * g1 + (1 + beta) * g2)

                gene1 = int(min(max(round(v1), 0), num_providers - 1))
                gene2 = int(min(max(round(v2), 0), num_providers - 1))

                if gene1 != i and gene2 != i:
                    break
                attempt += 1

            if gene1 == i:
                gene1 = (i + 1) % num_providers
            if gene2 == i:
                gene2 = (i + 2) % num_providers if (i + 1) % num_providers == gene1 else (i + 1) % num_providers

            child1[i], child2[i] = gene1, gene2
        else:
            gene1 = g1 if g1 != i else (i + 1) % num_providers
            gene2 = g2 if g2 != i else (i + 2) % num_providers if (i + 1) % num_providers == gene1 else (i + 1) % num_providers
            child1[i], child2[i] = gene1, gene2

    # spatial feasibility repair using rng
    for i in range(len(child1)):
        c = cons[i]
        if euclidean_distance(provs[child1[i]], c) > COVERAGE_RADIUS:
            valid = [j for j, p in enumerate(provs) if euclidean_distance(p, c) <= COVERAGE_RADIUS]
            if valid:
                child1[i] = int(rng.choice(valid))
    for i in range(len(child2)):
        c = cons[i]
        if euclidean_distance(provs[child2[i]], c) > COVERAGE_RADIUS:
            valid = [j for j, p in enumerate(provs) if euclidean_distance(p, c) <= COVERAGE_RADIUS]
            if valid:
                child2[i] = int(rng.choice(valid))

    return Individual(child1), Individual(child2)


def polynomial_mutation(ind: Individual,
                        eta: float,
                        pm: float,
                        curr_time,
                        rng) -> Individual:
    provs, cons = record_list_dict[curr_time]
    genes = ind.genes[:]
    if rng.random() > pm:
        return Individual(genes)

    i = int(rng.integers(0, len(genes)))
    x = genes[i]
    max_attempts = 10
    attempt = 0
    x_new = x
    c = cons[i]

    # quick feasibility repair
    if euclidean_distance(provs[x_new], c) > COVERAGE_RADIUS:
        valid = [j for j, p in enumerate(provs) if euclidean_distance(p, c) <= COVERAGE_RADIUS]
        if valid:
            x_new = int(rng.choice(valid))

    while attempt < max_attempts:
        u = rng.random()
        delta = (2 * u) ** (1.0 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
        x_new = x + int(round(delta * (num_providers - 1)))
        x_new = min(max(x_new, 0), num_providers - 1)
        if x_new != i:
            break
        attempt += 1

    if x_new == i:
        eligible = [j for j in range(num_providers) if j != i]
        x_new = int(rng.choice(eligible)) if eligible else i

    genes[i] = x_new
    return Individual(genes)


def select_diverse_leaders(archive: List[Tuple[np.ndarray, Tuple[float, float]]], 
                           target_count: int = 3) -> List[np.ndarray]:
    """
    Selects a diverse set of leader positions from the archive using Hamming distance.
    
    Args:
        archive (List[Tuple[np.ndarray, Tuple[float, float]]]): Archive of (position vector, (error, cost)).
        target_count (int): Number of diverse leaders to select.
    
    Returns:
        List[np.ndarray]: Selected leader position vectors.
    """
    if not archive:
        return []

    positions, _ = zip(*archive)
    positions = list(positions)
    
    # Start with the best (e.g., lowest error + cost) solution
    selected = [positions[0]]  

    while len(selected) < target_count and len(selected) < len(positions):
        candidates = [p for p in positions if not any(np.array_equal(p, s) for s in selected)]
        best_candidate = None
        best_min_dist = -1

        for candidate in candidates:
            # Min Hamming distance to current selected
            min_dist = min(hamming_distance(candidate, s) for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate

        if best_candidate is not None:
            selected.append(best_candidate)
        else:
            break  # fallback — no good new leader

    return selected


def mopso_algorithm_proper(
    type_error: str,
    swarm_size: int = 100,
    max_iterations: int = 120,
    archive_size: int = 120,
    w_max: float = 0.9,
    w_min: float = 0.4,
    c1: float = 1.8,
    c2: float = 1.8,
    v_max: float = 4.0
) -> Tuple[List[float], List[float], List[float]]:

    result_errors, result_costs, result_stds = [], [], []
    algorithm = "mopso"

    err_func_map = {'tp': QoS_tp_error, 'res': QoS_res_error, 'rel': relative_error}
    err_func = err_func_map.get(type_error, relative_error)

    for t in range(num_times):
        rng = rng_for('pso', t)  # <<< per-time RNG

        prods, cons = record_list_dict[t]
        D = len(cons)
        curr_max_cost = max(cost_per_dict[f"{t}"])
        curr_min_cost = min(cost_per_dict[f"{t}"])

        # Precompute QoS and cost matrices
        qos_matrix = np.zeros((num_providers, D), dtype=np.float32)
        cost_matrix = np.zeros((num_providers, D), dtype=np.float32)
        for p in range(num_providers):
            for ci, c in enumerate(cons):
                if euclidean_distance(prods[p], c) <= COVERAGE_RADIUS:
                    qos_matrix[p, ci] = norm_err(type_error, err_func(prods[p], c, algorithm=algorithm, time=t), t)
                    cost_matrix[p, ci] = (prods[p]['cost'] - curr_min_cost) / (curr_max_cost - curr_min_cost + 1e-12)
                    cost_matrix[p, ci] *= (1.0 + LAMBDA_VOL * prods[p].get('qos_volatility',0.0)) / (max(prods[p].get('qos_prob',0.5),1e-6)**GAMMA_QOS)
                else:
                    qos_matrix[p, ci] = 1.0
                    cost_matrix[p, ci] = 1.0

        # --- RNG-based initialization
        positions = rng.uniform(0, num_providers - 1, (swarm_size, D)).astype(np.float32)
        velocities = rng.uniform(-v_max, v_max, (swarm_size, D)).astype(np.float32)

        pbest_positions = positions.copy()
        pbest_objectives = np.full((swarm_size, 2), np.inf, dtype=np.float32)
        archive: List[Tuple[np.ndarray, Tuple[float, float]]] = []

        for iteration in range(max_iterations):
            w = w_max - (w_max - w_min) * (iteration / max(1, max_iterations - 1))

            # discretize and enforce spatial feasibility
            discrete_positions = np.clip(np.round(positions).astype(int), 0, num_providers - 1)
            for i in range(swarm_size):
                for j in range(D):
                    c = cons[j]
                    current_p = prods[discrete_positions[i, j]]
                    if euclidean_distance(current_p, c) > COVERAGE_RADIUS:
                        valid_providers = [pi for pi, p in enumerate(prods) if euclidean_distance(p, c) <= COVERAGE_RADIUS]
                        if valid_providers:
                            discrete_positions[i, j] = int(rng.choice(valid_providers))  # <<< rng
                        else:
                            raise RuntimeError(f"No valid provider within coverage for consumer {c['service_id']} at t={t}")

            # evaluate particles
            particle_indices = np.arange(D)
            all_errors = qos_matrix[discrete_positions, particle_indices]
            all_costs  = cost_matrix[discrete_positions, particle_indices]
            mean_errors = all_errors.mean(axis=1)
            mean_costs  = all_costs.mean(axis=1)
            current_objectives = np.stack((mean_errors, mean_costs), axis=1)

            # update personal bests
            for i in range(swarm_size):
                if dominates_objectives_mopso(tuple(current_objectives[i]), tuple(pbest_objectives[i])):
                    pbest_objectives[i] = current_objectives[i]
                    pbest_positions[i] = positions[i].copy()

            # update archive (non-dominated + crowding)
            current_solutions = [(positions[i].copy(), tuple(current_objectives[i])) for i in range(swarm_size)]
            all_solutions = archive + current_solutions
            all_objectives = [obj for _, obj in all_solutions]
            fronts = nondominated_fronts_mopso(all_objectives)
            if fronts:
                first_front_solutions = [all_solutions[i] for i in fronts[0]]
                if len(first_front_solutions) <= archive_size:
                    archive = first_front_solutions
                else:
                    front_objectives = [obj for _, obj in first_front_solutions]
                    cd = crowding_distance_mopso(front_objectives)
                    top_idx = np.argpartition(-np.array(cd), archive_size)[:archive_size]
                    archive = [first_front_solutions[i] for i in top_idx]

            # choose leaders (deterministic diversity heuristic), then RNG pick one per particle
            if archive:
                leaders = select_diverse_leaders(archive, target_count=3)
            else:
                leaders = [pbest_positions[i] for i in range(min(3, swarm_size))]

            for i in range(swarm_size):
                gbest_position = leaders[int(rng.integers(0, len(leaders)))]  # <<< rng

                r1 = rng.random(D)  # <<< rng
                r2 = rng.random(D)  # <<< rng
                cognitive = c1 * r1 * (pbest_positions[i] - positions[i])
                social    = c2 * r2 * (gbest_position   - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                positions[i]  = np.clip(positions[i] + velocities[i], 0, num_providers - 1)

        if archive:
            objs = [obj for _, obj in archive]
            fronts = nondominated_fronts_mopso(objs)
            record_front('mopso', type_error, t, [objs[i] for i in fronts[0]])
        else:
            record_front('mopso', type_error, t, [])


        # finalize one solution per t (use archive)
        if archive:
            best_solution_idx = int(np.argmin([obj[0] for _, obj in archive]))
            best_position_array = np.round(archive[best_solution_idx][0]).astype(int)

            # SCS bookkeeping
            for ci, producer_idx in enumerate(best_position_array):
                mapping = (prods[producer_idx], cons[ci])
                QoS_res_error(algorithm=algorithm, time=t, p=mapping[0], c=mapping[1], final=True)
                QoS_tp_error(algorithm=algorithm, time=t, p=mapping[0], c=mapping[1], final=True)
                update_streaks_with_marker(
                    alg=algorithm, metric=type_error, t=t,
                    c_id=SID_to_c_s_id(mapping[1]["service_id"]),
                    p_id=SID_to_c_s_id(mapping[0]["service_id"])
                )

            final_objectives = np.array([obj for _, obj in archive])
            result_errors.append(float(final_objectives[:, 0].mean()))
            result_costs.append(float(final_objectives[:, 1].mean()))
            result_stds.append(float(final_objectives[:, 0].std()) if len(final_objectives) > 1 else 0.0)
        else:
            result_errors.append(0.0); result_costs.append(0.0); result_stds.append(0.0)

    return result_errors, result_costs, result_stds


def nondominated_fronts_mopso(objectives):
    """
    Fast non-dominated sorting algorithm.
    Returns list of fronts, where each front is a list of indices.
    """
    n = len(objectives)
    if n == 0:
        return [[]]
    
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]
    
    # Calculate domination relationships
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates_objectives_mopso(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                elif dominates_objectives_mopso(objectives[j], objectives[i]):
                    domination_count[i] += 1
        
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Build subsequent fronts
    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        
        if next_front:
            fronts.append(next_front)
        current_front += 1
    
    return [front for front in fronts if front]


def dominates_objectives_mopso(obj1, obj2):
    """Check if obj1 dominates obj2 (minimization problem)."""
    better_in_at_least_one = False
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:  # obj1 is worse in this objective
            return False
        elif obj1[i] < obj2[i]:  # obj1 is better in this objective
            better_in_at_least_one = True
    return better_in_at_least_one


def crowding_distance_mopso(objectives):
    """
    Calculate crowding distance for a set of objectives.
    Returns list of crowding distances.
    """
    n = len(objectives)
    if n == 0:
        return []
    if n <= 2:
        return [float('inf')] * n
    
    distances = [0.0] * n
    n_obj = len(objectives[0]) if objectives else 0
    
    for m in range(n_obj):
        # Sort by m-th objective
        sorted_indices = sorted(range(n), key=lambda i: objectives[i][m])
        
        # Set boundary points to infinity
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # Calculate crowding distance for intermediate points
        obj_range = objectives[sorted_indices[-1]][m] - objectives[sorted_indices[0]][m]
        if obj_range > 0:
            for i in range(1, n-1):
                distances[sorted_indices[i]] += (
                    objectives[sorted_indices[i+1]][m] - objectives[sorted_indices[i-1]][m]
                ) / obj_range
    
    return distances


def hamming_distance(arr1, arr2):
    return np.sum(np.array(arr1) != np.array(arr2))

def select_fallback_leaders(leaders, prev_archive, target_count=3):
    """
    Fills up the `leaders` list with the most diverse candidates from `prev_archive`
    using Hamming distance to enforce diversity.
    """
    if len(leaders) >= target_count or not prev_archive:
        return leaders

    combinations, _ = zip(*prev_archive)
    candidates = [c for c in combinations if c not in leaders]
    
    while len(leaders) < target_count and candidates:
        # Compute min Hamming distance to existing leaders
        scored = []
        for cand in candidates:
            dists = [hamming_distance(cand, l) for l in leaders]
            min_dist = min(dists) if dists else float('inf')
            scored.append((min_dist, cand))

        # Choose candidate with max min-distance to current leaders
        scored.sort(reverse=True)  # Sort by distance descending
        _, best_candidate = scored[0]
        leaders.append(best_candidate)
        candidates.remove(best_candidate)
    
    return leaders

def select_alpha_beta_delta(archive: List[Tuple[np.ndarray, Tuple[float, float]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select alpha, beta, delta from the archive using dominance and crowding.
    Returns three leader positions as (alpha, beta, delta).
    """
    if len(archive) < 3:
        return archive[0][0], archive[0][0], archive[0][0]

    objectives = [obj for _, obj in archive]
    n = len(objectives)

    # Compute crowding distances
    distances = np.zeros(n)
    for i in range(2):
        sorted_idx = np.argsort([obj[i] for obj in objectives])
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = float('inf')
        for j in range(1, n - 1):
            dist = (objectives[sorted_idx[j + 1]][i] - objectives[sorted_idx[j - 1]][i]) + 1e-12
            distances[sorted_idx[j]] += dist

    # Sort by crowding distance descending
    sorted_indices = np.argsort(distances)[::-1]
    alpha = archive[sorted_indices[0]][0]
    beta  = archive[sorted_indices[1]][0]
    delta = archive[sorted_indices[2]][0]

    return alpha, beta, delta

def mogwo_algorithm_proper(
    type_error: str,
    wolf_size: int = 100,
    max_iters: int = 150,
    archive_size: int = 120,
) -> Tuple[List[float], List[float], List[float]]:

    result_errors, result_costs, result_stds = [], [], []
    algorithm = "mogwo"

    err_func_map = {'tp': QoS_tp_error, 'res': QoS_res_error, 'rel': relative_error}
    err_func = err_func_map.get(type_error, relative_error)

    for t in range(num_times):
        rng = rng_for('gwo', t)  # <<< per-time RNG

        prods, cons = record_list_dict[t]
        D = len(cons)
        curr_max_cost = max(cost_per_dict[f"{t}"])
        curr_min_cost = min(cost_per_dict[f"{t}"])

        # QoS & cost matrices
        qos_matrix = np.zeros((num_providers, D), dtype=np.float32)
        cost_matrix = np.zeros((num_providers, D), dtype=np.float32)
        for p in range(num_providers):
            for ci, c in enumerate(cons):
                if euclidean_distance(prods[p], c) <= COVERAGE_RADIUS:
                    qos_matrix[p, ci] = norm_err(type_error, err_func(prods[p], c, algorithm=algorithm, time=t), t)
                    cost_matrix[p, ci] = (prods[p]['cost'] - curr_min_cost) / (curr_max_cost - curr_min_cost + 1e-12)
                    cost_matrix[p, ci] *= (1.0 + LAMBDA_VOL * prods[p].get('qos_volatility',0.0)) / (max(prods[p].get('qos_prob',0.5),1e-6)**GAMMA_QOS)
                else:
                    qos_matrix[p, ci] = 1.0
                    cost_matrix[p, ci] = 1.0

        # --- RNG-based initialization
        wolves = rng.uniform(0, num_providers - 1, (wolf_size, D)).astype(np.float32)
        archive: List[Tuple[np.ndarray, Tuple[float, float]]] = []

        for iteration in range(max_iters):
            discrete_wolves = np.clip(np.round(wolves).astype(int), 0, num_providers - 1)

            # enforce spatial feasibility
            for i in range(wolf_size):
                for j in range(D):
                    c = cons[j]
                    current_p = prods[discrete_wolves[i, j]]
                    if euclidean_distance(current_p, c) > COVERAGE_RADIUS:
                        valid_providers = [pi for pi, p in enumerate(prods) if euclidean_distance(p, c) <= COVERAGE_RADIUS]
                        if valid_providers:
                            discrete_wolves[i, j] = int(rng.choice(valid_providers))  # <<< rng
                        else:
                            raise RuntimeError(f"No valid provider within coverage for consumer {c['service_id']} at t={t}")

            rows = discrete_wolves
            cols = np.tile(np.arange(D), (wolf_size, 1))
            errs  = qos_matrix[rows, cols]
            costs = cost_matrix[rows, cols]
            mean_errs  = errs.mean(axis=1)
            mean_costs = costs.mean(axis=1)
            wolf_objectives = list(zip(mean_errs.tolist(), mean_costs.tolist()))

            current_solutions = list(zip(wolves.copy(), wolf_objectives))
            all_solutions = archive + current_solutions
            all_objectives = [obj for _, obj in all_solutions]
            fronts = nondominated_fronts(all_objectives)
            if fronts:
                first_front = [all_solutions[i] for i in fronts[0]]
                if len(first_front) <= archive_size:
                    archive = first_front
                else:
                    front_objectives = [obj for _, obj in first_front]
                    cd = crowding_distance(front_objectives)
                    top_idx = np.argpartition(-np.array(cd), archive_size)[:archive_size]
                    archive = [first_front[i] for i in top_idx]

            if len(archive) >= 3:
                archive_positions, archive_objectives = zip(*archive)
                cd = crowding_distance(list(archive_objectives))
                sorted_indices = np.argsort(cd)[::-1]
                alpha_pos = archive_positions[sorted_indices[0]]
                beta_pos  = archive_positions[sorted_indices[1]]
                delta_pos = archive_positions[sorted_indices[2]]
            else:
                alpha_pos, beta_pos, delta_pos = select_alpha_beta_delta(archive)

            a = 2 - 2 * iteration / max_iters
            r = lambda: rng.random(D)  # <<< rng for all random vectors

            wolves_update = []
            for i in range(wolf_size):
                A1, C1 = 2 * a * r() - a, 2 * r()
                A2, C2 = 2 * a * r() - a, 2 * r()
                A3, C3 = 2 * a * r() - a, 2 * r()

                D_alpha = np.abs(C1 * alpha_pos - wolves[i])
                X1 = alpha_pos - A1 * D_alpha

                D_beta = np.abs(C2 * beta_pos - wolves[i])
                X2 = beta_pos - A2 * D_beta

                D_delta = np.abs(C3 * delta_pos - wolves[i])
                X3 = delta_pos - A3 * D_delta

                new_pos = (X1 + X2 + X3) / 3
                wolves_update.append(np.clip(new_pos, 0, num_providers - 1))

            wolves = np.array(wolves_update, dtype=np.float32)

        if archive:
            objs = [obj for _, obj in archive]
            fronts = nondominated_fronts(objs)
            record_front('mogwo', type_error, t, [objs[i] for i in fronts[0]])
        else:
            record_front('mogwo', type_error, t, [])


        if archive:
            best_idx = int(np.argmin([obj[0] for _, obj in archive]))
            best_position = np.round(archive[best_idx][0]).astype(int)

            for ci, producer_idx in enumerate(best_position):
                mapping = (prods[producer_idx], cons[ci])
                QoS_res_error(p=mapping[0], c=mapping[1], time=t, algorithm=algorithm, final=True)
                QoS_tp_error(p=mapping[0], c=mapping[1], time=t, algorithm=algorithm, final=True)
                update_streaks_with_marker(
                    alg=algorithm, metric=type_error, t=t,
                    c_id=SID_to_c_s_id(mapping[1]["service_id"]),
                    p_id=SID_to_c_s_id(mapping[0]["service_id"])
                )

            final_objectives = np.array([obj for _, obj in archive])
            result_errors.append(float(final_objectives[:, 0].mean()))
            result_costs.append(float(final_objectives[:, 1].mean()))
            result_stds.append(float(final_objectives[:, 0].std()) if len(final_objectives) > 1 else 0.0)
        else:
            result_errors.append(0.0); result_costs.append(0.0); result_stds.append(0.0)

    return result_errors, result_costs, result_stds


def nondominated_fronts(objectives):
    """
    Fast non-dominated sorting algorithm.
    Returns list of fronts, where each front is a list of indices.
    """
    n = len(objectives)
    if n == 0:
        return [[]]
    
    domination_count = [0] * n  # Number of solutions that dominate solution i
    dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by solution i
    fronts = [[]]
    
    # Calculate domination relationships
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates_objectives(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                elif dominates_objectives(objectives[j], objectives[i]):
                    domination_count[i] += 1
        
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Build subsequent fronts
    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        
        if next_front:
            fronts.append(next_front)
        current_front += 1
    
    # Remove empty fronts
    return [front for front in fronts if front]


def dominates_objectives(obj1, obj2):
    """Check if obj1 dominates obj2 (minimization problem)."""
    better_in_at_least_one = False
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:  # obj1 is worse in this objective
            return False
        elif obj1[i] < obj2[i]:  # obj1 is better in this objective
            better_in_at_least_one = True
    return better_in_at_least_one


def crowding_distance_mogwo(objectives):
    """
    Calculate crowding distance for a set of objectives.
    Returns list of crowding distances.
    """
    n = len(objectives)
    if n == 0:
        return []
    if n <= 2:
        return [float('inf')] * n
    
    distances = [0.0] * n
    n_obj = len(objectives[0]) if objectives else 0
    
    for m in range(n_obj):
        # Sort by m-th objective
        sorted_indices = sorted(range(n), key=lambda i: objectives[i][m])
        
        # Set boundary points to infinity
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # Calculate crowding distance for intermediate points
        obj_range = objectives[sorted_indices[-1]][m] - objectives[sorted_indices[0]][m]
        if obj_range > 0:
            for i in range(1, n-1):
                distances[sorted_indices[i]] += (
                    objectives[sorted_indices[i+1]][m] - objectives[sorted_indices[i-1]][m]
                ) / obj_range
    
    return distances


def plot_cost_comparison(times, costs, labels, title="Cost Comparison"):
    """Plot cost comparison between algorithms"""
    plt.figure(figsize=(12, 6))
    
    for i, (cost, label) in enumerate(zip(costs, labels)):
        plt.plot(times, cost, marker='s', label=label)
    
    plt.xlabel("Time Frame")
    plt.ylabel("Cost")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    #eturn plt.gcf()

def plot_cost_error_trade_off(errors, costs, labels, title="Cost-Error Trade-off"):
    """Plot cost vs error trade-off"""
    plt.figure(figsize=(10, 8))
    
    markers = ['o', 'x', 'd']
    for i, (err, cost, label) in enumerate(zip(errors, costs, labels)):
        plt.scatter(err, cost, marker=markers[i], label=label, s=50)
    
    plt.xlabel("Mean Error")
    plt.ylabel("Mean Cost")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    
def pareto_frontier(xs, ys):
    """Return boolean mask of Pareto‐optimal points (minimize both xs and ys)."""
    is_pareto = np.ones(len(xs), dtype=bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if is_pareto[i]:
            # any point strictly better in both dims?
            is_pareto[is_pareto] = np.any(
                np.vstack([xs[is_pareto] < x, ys[is_pareto] < y]), axis=0
            )
            is_pareto[i] = True
    return is_pareto

def get_maxmin():
    all_tp_errs, all_res_errs = [], []
    per_time_obj, max_min_obj = {}, {}

    for t in range(num_times):
        per_time_tp, per_time_res = [], []
        prods, cons = record_list_dict[t]

        for p in prods:
            for c in cons:
                # <-- only consider feasible links
                if euclidean_distance(p, c) > COVERAGE_RADIUS:
                    continue
                e_tp  = reg_err(p, c, "tp")
                e_res = reg_err(p, c, "res")
                all_tp_errs.append(e_tp)
                all_res_errs.append(e_res)
                per_time_tp.append(e_tp)
                per_time_res.append(e_res)

        if not per_time_tp or not per_time_res:
            raise RuntimeError(
                f"No feasible provider–consumer pairs at t={t} within radius {COVERAGE_RADIUS}."
            )

        per_time_obj[f"{t}"] = [per_time_tp, per_time_res]
        max_min_obj[f"{t}"] = [
            max(per_time_tp), max(per_time_res),
            min(per_time_tp), min(per_time_res)
        ]

    return all_tp_errs, all_res_errs, per_time_obj, max_min_obj


# --- MAIN & EVALUATION ---------------------------------------
if __name__ == "__main__":
    #initiate_records()
    types = ['tp', 'res']
    times = list(range(num_times))
    
    all_runs_metrics = []
   
    k_pso = 0.2
    # Before: you were appending only the [mean_err, mean_cost, err_std] triples.
# Instead, build dicts mapping each error-type to a 2D list:

# Initialize containers
    greedy_series = { t: {'errors': [], 'costs': [], 'stds': []} for t in types }
    random_series = { t: {'errors': [], 'costs': [], 'stds': []} for t in types }
    nsga_series   = { t: {'errors': [], 'costs': [], 'stds': []} for t in types }
    pso_series = { t: {'errors':[], 'costs':[], 'stds':[]} for t in types }
    mogwo_series = {t: {'errors': [], 'costs':[], 'stds': []} for t in types}

    for i in range(num_iterations):
        # ensure reproducibility if you want
        dataset_type = 'first'
        front_log.clear()

        tp_list = [x for x in tp_two if float(x) >= 0] if dataset_type=='second' else [x for x in service_tp_list if float(x) >= 0]
        rt_list = [x for x in rs_two if float(x) >= 0] if dataset_type=='second' else [x for x in service_rt_list if float(x) >= 0]
        num_services_total = len(service_ids)

        provider_qos_raw = collect_raw_qos_sequences(
            rt_list, tp_list, num_times, num_services_total, num_providers
        )

        transition_matrix = construct_transition_matrix(provider_qos_raw)

        initiate_records(rt_list, tp_list, transition_matrix, num_times, num_services_total, num_providers) # used this for testing -- when done testing remove random seed.
        
        # After `initiate_records()` but before any greedy/nsga/pso calls:
        

        all_tp_errs, all_res_errs, per_time_obj, max_min_object = get_maxmin()

        # Min-max bounds for [0,1] scaling

        global max_min_obj
        global per_time_object 

        max_min_obj = max_min_object
        per_time_object = per_time_obj
                    
        global tp_min, tp_max, res_min, res_max, max_tp_err, max_res_err

        tp_min, tp_max = min(all_tp_errs), max(all_tp_errs)
        res_min, res_max = min(all_res_errs), max(all_res_errs)

        # (Optional) z-score stats
        tp_mean, tp_std = np.mean(all_tp_errs), np.std(all_tp_errs)
        res_mean, res_std = np.mean(all_res_errs), np.std(all_res_errs)


        max_tp_err = tp_max
        max_res_err = res_max
        # Run one batch of experiments
        for t in types:
            errs, costs, stds = greedy_algorithm(t, record_cb=record_front)

            greedy_series[t]['errors'].append(errs)
            greedy_series[t]['costs'].append(costs)
            greedy_series[t]['stds'].append(stds)
            
            # #errs, costs, stds = random_algorithm(t)
            # random_series[t]['errors'].append(errs)
            # random_series[t]['costs'].append(costs)
            # random_series[t]['stds'].append(stds)

          
            
            errs, costs, stds = mopso_algorithm_proper(t)
            pso_series[t]['errors'].append(errs)
            pso_series[t]['costs'].append(costs)
            pso_series[t]['stds'].append(stds)

         

            mogwo_err, mogwo_cost, mogwo_std = mogwo_algorithm_proper(t)
            mogwo_series[t]['errors'].append(mogwo_err)
            mogwo_series[t]['costs'].append(mogwo_cost)
            mogwo_series[t]['stds'].append(mogwo_std)
            
            
            # For NSGA, we need to collect per‐time values, so:
            iter_nsga_err = []
            iter_nsga_cost = []
            iter_nsga_std = []
            weight = 0.05 # try this... allows for some exploitation but also preserves diversity for future times
            for ti in range(num_times):
                _, mean_err, mean_cost, std_err = enhanced_NSGA_II_for_time(ti, t, weight)
                iter_nsga_err.append(mean_err)
                iter_nsga_cost.append(mean_cost)
                iter_nsga_std.append(std_err)
            nsga_series[t]['errors'].append(iter_nsga_err)
            nsga_series[t]['costs'].append(iter_nsga_cost)
            nsga_series[t]['stds'].append(iter_nsga_std)
        
        run_metrics = compute_indicators_for_run()
        all_runs_metrics.append(run_metrics)

        # one run only
        ind = all_runs_metrics[0]
        times = list(range(num_times))
        for te, title in [('tp','Throughput'), ('res','Response-time')]:
            plt.figure(figsize=(10,4))
            for alg in ['nsga','mopso','mogwo','greedy']:
                plt.plot(times, ind[alg][te]['HV'],  marker='o', label=f'{alg.upper()} HV')
            plt.ylim(0,1); plt.title(f'Hypervolume over time ({title})'); plt.xlabel('t'); plt.ylabel('HV'); plt.grid(True); plt.legend(); plt.show()

            plt.figure(figsize=(10,4))
            for alg in ['nsga','mopso','mogwo','greedy']:
                plt.plot(times, ind[alg][te]['IGD'], marker='s', label=f'{alg.upper()} IGD')
            plt.title(f'IGD over time ({title})'); plt.xlabel('t'); plt.ylabel('IGD (↓ better)'); plt.grid(True); plt.legend(); plt.show()

            plt.figure(figsize=(10,4))
            for alg in ['nsga','mopso','mogwo','greedy']:
                plt.plot(times, ind[alg][te]['EPS'], marker='^', label=f'{alg.upper()} ε')
            plt.title(f'Epsilon over time ({title})'); plt.xlabel('t'); plt.ylabel('ε (↓ better)'); plt.grid(True); plt.legend(); plt.show()


        # Prepare final aggregated metrics
        agg = {'greedy': {}, 'random': {}, 'nsga': {}, 'PSO': {}, 'mogwo':{}}
        for name, series in [('greedy', greedy_series), ('random', random_series), ('nsga', nsga_series), ('PSO', pso_series), ('mogwo', mogwo_series)]:
            for t in types:
                errs_arr = np.array(series[t]['errors'])   # shape: (num_iterations, num_times)
                costs_arr= np.array(series[t]['costs'])
                stds_arr = np.array(series[t]['stds'])
                

                agg[name].setdefault('mean_err', {})[t]  = errs_arr.mean(axis=0).tolist()
                agg[name].setdefault('mean_cost',{})[t]  = costs_arr.mean(axis=0).tolist()
                agg[name].setdefault('err_std_band',{})[t] = stds_arr.mean(axis=0).tolist()  # if you want the average per-time std
                
                # If you prefer plotting a confidence band you could also store the across-iteration std:
                agg[name].setdefault('iter_std',{})[t] = errs_arr.std(axis=0).tolist()

        times = list(range(num_times))

    
    avg_SCS_time_series_greedy_tp = []
    avg_SCS_time_series_nsga_tp = []
    avg_SCS_time_series_mopso_tp = []
    avg_SCS_time_series_mogwo_tp = []

    avg_SCS_time_series_greedy_res = []
    avg_SCS_time_series_nsga_res = []
    avg_SCS_time_series_mopso_res = []
    avg_SCS_time_series_mogwo_res = []



    print("Series data for MOGWO is empty: {}".format(len(agg['mogwo']['iter_std']['tp']) == 0))


    for te, label in [('tp','Throughput Error Std'), ('res','Response-Time Error Std')]:
        plt.figure(figsize=(10,5))
        plt.plot(times, agg['nsga']['iter_std'][te],   marker='o', label=f'NSGA-II')
        plt.plot(times, agg['greedy']['iter_std'][te], marker='x', label=f'Greedy {label}')
        #plt.plot(times, agg['random']['iter_std'][te], marker='d', label=f'Random {label}')
        plt.plot(times, agg['PSO']['iter_std'][te], marker = 's', label=f'PSO')
        plt.plot(times, agg['mogwo']['iter_std'][te], marker = '*', label = f'MOGWO')
        plt.xlabel("Time Frame")
        plt.ylabel("Std. Dev. of Error")
        plt.title(f"{label} over {num_times} Time Frames")
        plt.legend()
        plt.grid(True)

        plt.show()


    # --- 2. MEAN ERROR LINE PLOTS -------------------------------

    for te, label in [('tp','Throughput Error'), ('res','Response-Time Error')]:
        plt.figure(figsize=(10,5))
        plt.plot(times, agg['nsga']['mean_err'][te],   marker='o', label=f'NSGA-II {label}')
        plt.plot(times, agg['greedy']['mean_err'][te], marker='x', label=f'Greedy {label}')
        #plt.plot(times, agg['random']['mean_err'][te], marker='d', label=f'Random {label}')
        plt.plot(times, agg['PSO']['mean_err'][te], marker = 'o', label=f'PSO')
        plt.plot(times, agg['mogwo']['mean_err'][te], marker = '*', label = f'MOGWO')
        plt.xlabel("Time Frame")
        plt.ylabel("Mean Error")
        plt.title(f"{label} Comparison over {num_times} Time Frames")
        plt.legend()
        plt.grid(True)
        plt.show()


    # --- 3. MEAN COST LINE PLOTS --------------------------------

    # You can reuse 'tp' label for throughput cost, and 'res' for response cost
    for te, label in [('tp','Throughput Cost'), ('res','Response-Time Cost')]:
        plt.figure(figsize=(10,5))
        plt.plot(times, agg['nsga']['mean_cost'][te],   marker='o', label=f'NSGA-II {label}')
        plt.plot(times, agg['greedy']['mean_cost'][te], marker='x', label=f'Greedy {label}')
        #plt.plot(times, agg['random']['mean_cost'][te], marker='d', label=f'Random {label}')
        plt.plot(times, agg['PSO']['mean_cost'][te], marker = 'o', label=f'PSO')
        plt.plot(times, agg['mogwo']['mean_cost'][te], marker = '*', label=f'MOGWO')
        plt.xlabel("Time Frame")
        plt.ylabel("Mean Cost")
        plt.title(f"{label} Comparison over {num_times} Time Frames")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    for metric_type, metric_dict in [
    ('Cost', 'mean_cost'),
    ('Error', 'mean_err')]:
        plt.figure(figsize=(10, 6))
        for alg, marker in [('mogwo', '*')]:
            # extract arrays
            x = np.array(agg[alg][metric_dict]['tp'])  # throughput
            y = np.array(agg[alg][metric_dict]['res'])  # latency
            # compute Pareto mask
            mask = pareto_frontier(x, y)
            
            # scatter all points
            plt.scatter(x, y, marker=marker, alpha=0.3, label=f'{alg.upper()} all')
            # highlight Pareto front
            plt.scatter(x[mask], y[mask], marker=marker, label=f'{alg.upper()} Pareto')
        
        plt.xlabel(f'Mean Throughput {metric_type}')
        plt.ylabel(f'Mean Latency {metric_type}')
        plt.title(f'Pareto Front: Throughput vs. Latency ({metric_type})')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    # --- 4. PARETO FRONT: ERROR vs COST for Throughput and Latency ---
    for te, label in [('tp', 'Throughput'), ('res', 'Response-Time')]:
        plt.figure(figsize=(10,6))
        plt.xscale('symlog', linthresh=1e-3)
        plt.yscale('symlog', linthresh=1e-3)
        # ('nsga','o'), ('greedy','s'), ('PSO', 'd'),
        for alg, marker in [('mogwo', '*'),('nsga','o'), ('greedy', 's'), ('PSO', 'd')]:
            # x = mean error for this type
            x = np.array(agg[alg]['mean_err'][te])
            # y = mean cost for same type
            y = np.array(agg[alg]['mean_cost'][te])
            
            # compute Pareto‐optimal mask (minimize both error and cost)
            mask = pareto_frontier(x, y)
            
            # plot all points lightly
            plt.scatter(x, y,
                        alpha=0.3,
                        marker=marker,
                        label=f"{alg.upper()} all")
            # highlight Pareto front
            plt.scatter(x[mask], y[mask],
                        alpha=1.0,
                        edgecolor='k',
                        s=80,
                        marker=marker,
                        label=f"{alg.upper()} Pareto")
            
        # all_x = np.concatenate([x for alg in ['nsga','greedy','random','PSO']
        #                 for x in [np.array(agg[alg]['mean_err'][te])]])
        # all_y = np.concatenate([y for alg in ['nsga','greedy','random','PSO']
        #                         for y in [np.array(agg[alg]['mean_cost'][te])]])

        plt.autoscale(enable=True, axis='both', tight=False)
        plt.margins(0.05)  # adds a 5% pad on all sides


        plt.xlabel(f"Mean {label} Error")
        plt.ylabel(f"Mean {label} Cost")
        plt.title(f"Pareto Front: {label} Error vs Cost")
        plt.legend()
        plt.grid(True)
        plt.show()

