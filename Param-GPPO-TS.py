import math
import random
import time
import os
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================== Configuration ===================
dataset_path    = "" # Path to the region-specific dataset (e.g., Area A/B/C)

ROADLEN_CSV  = ""      # Per-node intrinsic length: (id, length). Depot length is assumed to be 0.
DISTM_PATH   = ""  # Task-to-task empirical distance matrix
SEED = 42

M_SALESMEN = 5
LAMBDA_BAL = 0.15  # Weight of route-length dispersion (std) in the soft objective

# Graph construction for the GNN encoder (Euclidean KNN graph)
KNN_K = 12
GNN_HIDDEN = 96
GNN_LAYERS = 2

# PPO hyperparameters
PPO_GAMMA = 0.98
PPO_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_LR = 3e-4
PPO_EPOCHS = 3
PPO_BATCH = 128
PPO_MINI_BATCH = 64
PPO_ENTROPY = 1.0e-2
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 1.0

# PPO adaptive schedule (KL-based control)
KL_TARGET = 0.02
LR_DECAY = 0.9
LR_GROW  = 1.05
ENTROPY_MIN = 3e-3
ENTROPY_MAX = 2.0e-2

# Heuristic initialization and local search control
PRINT_EVERY = 20
INIT_SWEEP_STARTS = 12
INIT_TWO_OPT_ITERS = 500
INIT_OROPT_ROUNDS = 1

REINFORCE_2OPT_ITERS = 500
REINFORCE_OROPT_ROUNDS = 1

# Ruin-and-recreate configuration (Shaw-style relatedness ruin)
SHAW_RUIN_FRAC = 0.12
INTER_POS_SAMPLES = 32
TOP_CAND_NODES = 96

# Total number of decision steps (outer loop iterations)
TOTAL_STEPS = 3000

# ε-greedy exploration schedule
EPS0 = 0.15
EPS_MIN = 0.05
EPS_DECAY_STEPS = 1500

# ---- Lexicographic reward shaping and stabilization ----
DELTA_MAX_PRIMARY_THR_FRAC = 0.004
STD_WORSEN_MARGIN = 0.5
REWARD_CLIP_FRAC = 0.02
STD_CLIP_FRAC    = 0.05
EMA_BETA         = 0.9
SMALL_NEG_REWARD = -1e-3

# Acceptance threshold schedule (used to reject severely deteriorating moves)
ACCEPT_TAU_START = 0.04
ACCEPT_TAU_END   = 0.02

# Early-stage operator diversification
RR_EARLY_STEPS = 300

# Periodic intensification and diversification
INTENSIFY_EVERY  = 25
DIVERSIFY_EVERY  = 120
LIGHT_PERTURB_FRAC = 0.03

# Operator credit bias schedule (success-based operator preference)
OP_CREDIT_BETA_START = 0.2
OP_CREDIT_BETA_END   = 0.05
OP_CREDIT_DECAY      = 0.90

OPS = ["intra_2opt", "intra_oropt", "inter_relocate", "inter_swap", "ruin_recreate", "reinforce_longest"]
N_OPS = len(OPS)

# Output paths
OUTPUT_DIR = "HVCoS-GPPO-ParamTS-AreaX-outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROGRESS_CSV = os.path.join(OUTPUT_DIR, "routes_progress.csv")
BEST_JSON    = os.path.join(OUTPUT_DIR, "routes_best.json")
PPO_LOSS_CSV = os.path.join(OUTPUT_DIR, "ppo_losses.csv")
PPO_LOSS_PNG = os.path.join(OUTPUT_DIR, "ppo_loss_curve.png")
for p in [PROGRESS_CSV, BEST_JSON, PPO_LOSS_CSV, PPO_LOSS_PNG]:
    if os.path.exists(p):
        os.remove(p)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =================== Utilities ===================
def euclidian(c: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix for coordinates c.
    """
    a2 = np.sum(c**2, axis=1, keepdims=True)
    d2 = a2 + a2.T - 2 * c @ c.T
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2, dtype=np.float64)

def build_hybrid_distance(coords: np.ndarray,
                          orig_ids: list,
                          distm_csv: str) -> np.ndarray:
    """
    Build a hybrid distance matrix over (depot + tasks).

    Indexing:
      - Node 0 is the depot.
      - Nodes 1..N-1 are task nodes.

    Distances:
      - Task-to-task distances are read from distance_matrix.csv and aligned by original task IDs.
      - Depot-to-task distances (and vice versa) are computed via Euclidean distance.
    """
    N = len(coords)
    dist_all = np.zeros((N, N), dtype=np.float64)

    # Depot ↔ task distances via Euclidean metric
    eud = euclidian(coords)
    dist_all[0, :] = eud[0, :]
    dist_all[:, 0] = eud[:, 0]

    if not os.path.exists(distm_csv):
        raise FileNotFoundError(f"distance_matrix not found: {distm_csv}")

    df = pd.read_csv(distm_csv, sep=None, engine="python")
    df = df.set_index(df.columns[0])
    df.index  = df.index.astype(str)
    df.columns = df.columns.astype(str)

    # Map original task ID to internal index (1..N-1)
    id_to_internal = {}
    for internal_idx, oid in enumerate(orig_ids, start=1):
        id_to_internal[str(oid)] = internal_idx

    # Populate task-to-task distances from the empirical matrix
    for id_i, i_internal in id_to_internal.items():
        if id_i not in df.index:
            continue
        for id_j, j_internal in id_to_internal.items():
            if i_internal == j_internal:
                dist_all[i_internal, j_internal] = 0.0
                continue
            v1 = df.at[id_i, id_j] if id_j in df.columns else np.nan
            v2 = df.at[id_j, id_i] if (id_j in df.index and id_i in df.columns) else np.nan
            cand = []
            for v in (v1, v2):
                if pd.notna(v) and float(v) != 0.0:
                    cand.append(float(v))
            # Use a conservative symmetrization choice when both directions exist
            val = max(cand) if cand else 0.0
            dist_all[i_internal, j_internal] = val

    # Enforce symmetry and zero diagonal
    i_triu, j_triu = np.triu_indices(N, k=1)
    dist_all[j_triu, i_triu] = dist_all[i_triu, j_triu]
    np.fill_diagonal(dist_all, 0.0)
    return dist_all

def submatrix(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Extract a submatrix using index set idx.
    """
    return mat[np.ix_(idx, idx)]

# ----- Route length = edge length + node intrinsic length -----
def tour_length(route: List[int], dist: np.ndarray, node_len: np.ndarray) -> float:
    """
    Compute the total length of a depot-start-depot route under:
      length = sum(edge lengths) + sum(node intrinsic lengths over visited task nodes).
    The depot node length is excluded (assumed 0).
    """
    edge_sum = float(sum(dist[route[i], route[i+1]] for i in range(len(route)-1)))
    node_sum = float(sum(node_len[v] for v in route[1:-1]))  # exclude depot (0)
    return edge_sum + node_sum

def lengths_all(routes: List[List[int]], dist: np.ndarray, node_len: np.ndarray) -> List[float]:
    """
    Compute route lengths for all salespersons.
    """
    return [tour_length(r, dist, node_len) for r in routes]

def soft_obj_by_lengths(Ls: List[float]) -> float:
    """
    Soft objective: max(route_len) + lambda * std(route_len).
    """
    arr = np.array(Ls, dtype=np.float64)
    return float(arr.max() + LAMBDA_BAL * (arr.std() + 1e-12))

def compute_obj(routes: List[List[int]], dist: np.ndarray, node_len: np.ndarray) -> Tuple[float, float]:
    """
    Return (primary objective, soft objective).
      - Primary: max route length
      - Soft: max + lambda * std
    """
    Ls = lengths_all(routes, dist, node_len)
    return float(max(Ls)), soft_obj_by_lengths(Ls)

def angle_sweep_partition(coords: np.ndarray, m: int, theta0: float = 0.0) -> List[np.ndarray]:
    """
    Angle-sweep clustering around the depot to generate an initial partition into m groups.
    """
    depot = coords[0]
    vecs = coords[1:] - depot[None, :]
    ang = (np.arctan2(vecs[:, 1], vecs[:, 0]) - theta0 + 2*np.pi) % (2*np.pi)
    order = np.argsort(ang) + 1
    n = len(order)
    base = n // m
    rem = n % m
    clusters = []
    start = 0
    for k in range(m):
        sz = base + (1 if k < rem else 0)
        idx = order[start:start+sz]
        start += sz
        clusters.append(np.concatenate(([0], idx)))
    return clusters

def apply_2opt(route: List[int], i: int, j: int) -> List[int]:
    """
    Apply a 2-opt reversal between indices [i, j] (inclusive).
    """
    return route[:i] + route[i:j+1][::-1] + route[j+1:]

def two_opt_best_once(route: List[int], dist: np.ndarray) -> Tuple[List[int], float]:
    """
    Perform one best-improvement 2-opt move. Returns (new_route, gain).
    """
    best_gain, best_i, best_j = 0.0, None, None
    n = len(route)
    for i in range(1, n-2):
        a, b = route[i-1], route[i]
        dab = dist[a, b]
        for j in range(i+1, n-1):
            c, d = route[j], route[j+1]
            gain = (dab + dist[c, d]) - (dist[a, c] + dist[b, d])
            if gain > best_gain:
                best_gain, best_i, best_j = gain, i, j
    if best_gain > 1e-12:
        return apply_2opt(route, best_i, best_j), best_gain
    return route, 0.0

def two_opt_improve(route: List[int], dist: np.ndarray, iters: int = 200) -> List[int]:
    """
    Iteratively apply best-improvement 2-opt until convergence or max iterations.
    """
    r = route
    for _ in range(iters):
        r2, g = two_opt_best_once(r, dist)
        if g <= 1e-12:
            break
        r = r2
    return r

def or_opt_once(route: List[int], dist: np.ndarray, ks=(1, 2, 3)) -> Tuple[List[int], float]:
    """
    Perform one best-improvement Or-opt move (relocating a segment of length k).
    Returns (new_route, gain).
    """
    n = len(route)
    best_gain = 0.0
    best = None
    for k in ks:
        if n <= 2 + k:
            continue
        for i in range(1, n-1-k+1):
            seg = route[i:i+k]
            a = route[i-1]
            b = route[i+k]
            remove_cost = dist[a, route[i]] + dist[route[i+k-1], b]
            add_ab = dist[a, b]
            delta_remove = remove_cost - add_ab
            for j in range(1, n-k):
                if i <= j <= i+k-1:
                    continue
                u, v = route[j], route[j+1]
                delta_add = (dist[u, seg[0]] + dist[seg[-1], v]) - dist[u, v]
                gain = delta_remove - delta_add
                if gain > best_gain:
                    best_gain = gain
                    best = (i, k, j)
    if best and best_gain > 1e-12:
        i, k, j = best
        seg = route[i:i+k]
        r = route[:i] + route[i+k:]
        pos = (j+1) if j < i else (j-k+1)
        r = r[:pos] + seg + r[pos:]
        return r, best_gain
    return route, 0.0

def build_init(coords: np.ndarray, m: int, dist_all: np.ndarray, node_len: np.ndarray) -> List[List[int]]:
    """
    Construct an initial solution by:
      1) Angle-sweep partitioning (multiple starting angles),
      2) Greedy nearest-neighbor route within each cluster,
      3) Intra-route improvement (2-opt + Or-opt),
    and selecting the best solution under the soft objective.
    """
    best_routes, best_soft = None, float("inf")
    for s in range(INIT_SWEEP_STARTS):
        theta0 = 2*np.pi * s / INIT_SWEEP_STARTS
        clusters = angle_sweep_partition(coords, m, theta0)
        routes = []
        for sub in clusters:
            dsub = submatrix(dist_all, sub)  # use empirical edge distances during initialization
            r = [0]
            unv = set(range(1, len(sub)))
            cur = 0
            while unv:
                nxt = min(unv, key=lambda j: dsub[cur, j])
                r.append(nxt)
                unv.remove(nxt)
                cur = nxt
            r.append(0)
            r = two_opt_improve(r, dsub, iters=INIT_TWO_OPT_ITERS)
            for _ in range(INIT_OROPT_ROUNDS):
                r2, g = or_opt_once(r, dsub, ks=(1, 2, 3))
                if g <= 1e-12:
                    break
                r = r2
            routes.append([int(sub[k]) for k in r])
        _, soft = compute_obj(routes, dist_all, node_len)
        if soft < best_soft - 1e-12:
            best_routes, best_soft = routes, soft
    return best_routes

# ======= Cross-route operators =======
def top_contrib_nodes(route: List[int], dist: np.ndarray, k: int) -> List[int]:
    """
    Identify nodes with high removal gain (i.e., large contribution to route cost).
    Returns indices (positions) in the route list.
    """
    n = len(route)
    contrib = []
    for i in range(1, n-1):
        a, b, c = route[i-1], route[i], route[i+1]
        gain_remove = (dist[a, b] + dist[b, c]) - dist[a, c]
        contrib.append((gain_remove, i))
    contrib.sort(reverse=True)
    return [idx for _, idx in contrib[:min(k, len(contrib))]]

def try_inter_relocate(routes: List[List[int]], dist: np.ndarray, node_len: np.ndarray) -> List[List[int]]:
    """
    Heuristic inter-route relocate:
      - Select the current longest route,
      - Try relocating one high-contribution node into another route,
      - Evaluate candidates under the soft objective.
    """
    Ls = lengths_all(routes, dist, node_len)
    ridx = int(np.argmax(Ls))
    long_r = routes[ridx]
    if len(long_r) <= 3:
        return routes
    cand_idx = top_contrib_nodes(long_r, dist, TOP_CAND_NODES)
    best_soft = soft_obj_by_lengths(Ls)
    best = None
    for i in cand_idx:
        node = long_r[i]
        a, b, c = long_r[i-1], long_r[i], long_r[i+1]
        delta_remove_edges = (dist[a, c] - (dist[a, b] + dist[b, c]))
        delta_remove_total = delta_remove_edges - node_len[node]
        for t in range(len(routes)):
            if t == ridx:
                continue
            r2 = routes[t]
            positions = list(range(1, len(r2)))
            rng.shuffle(positions)
            positions = positions[:min(INTER_POS_SAMPLES, len(positions))]
            for pos in positions:
                u, v = r2[pos-1], r2[pos]
                delta_add_edges = (dist[u, node] + dist[node, v]) - dist[u, v]
                delta_add_total = delta_add_edges + node_len[node]
                newLs = Ls[:]
                newLs[ridx] = Ls[ridx] + delta_remove_total
                newLs[t]    = Ls[t]    + delta_add_total
                s = soft_obj_by_lengths(newLs)
                if s < best_soft - 1e-12:
                    best_soft = s
                    best = (ridx, i, t, pos)
    if best is None:
        return routes
    ridx, i, t, pos = best
    new_routes = [list(r) for r in routes]
    node = new_routes[ridx].pop(i)
    new_routes[t].insert(pos, node)
    return new_routes

def try_inter_swap(routes: List[List[int]], dist: np.ndarray, node_len: np.ndarray) -> List[List[int]]:
    """
    Heuristic inter-route swap:
      - Select the current longest route,
      - Swap a high-contribution node with a node in another route,
      - Evaluate using the soft objective.
    """
    Ls = lengths_all(routes, dist, node_len)
    ridx = int(np.argmax(Ls))
    long_r = routes[ridx]
    if len(long_r) <= 3:
        return routes
    cand_i = top_contrib_nodes(long_r, dist, TOP_CAND_NODES)
    best = None
    best_soft = soft_obj_by_lengths(Ls)
    for i in cand_i:
        nodeA = long_r[i]
        for t in range(len(routes)):
            if t == ridx:
                continue
            r2 = routes[t]
            if len(r2) <= 3:
                continue
            idxs = list(range(1, len(r2)-1))
            rng.shuffle(idxs)
            idxs = idxs[:min(INTER_POS_SAMPLES, len(idxs))]
            for j in idxs:
                nodeB = r2[j]
                new_routes = [list(r) for r in routes]
                new_routes[ridx][i] = nodeB
                new_routes[t][j] = nodeA
                _, s = compute_obj(new_routes, dist, node_len)
                if s < best_soft - 1e-12:
                    best_soft = s
                    best = new_routes
    return routes if best is None else best

def ruin_recreate(routes: List[List[int]], dist: np.ndarray, coords: np.ndarray,
                  node_len: np.ndarray, frac=0.12) -> List[List[int]]:
    """
    Ruin-and-recreate operator (Shaw-style relatedness):
      - Remove a fraction of nodes related to a seed node from the longest route,
      - Reinsert each removed node greedily by minimal insertion cost (edge + node length).
    """
    all_nodes = []
    for r in routes:
        all_nodes.extend(r[1:-1])
    n = len(all_nodes)
    ruin_n = max(2, int(frac * n))

    Ls = lengths_all(routes, dist, node_len)
    ridx = int(np.argmax(Ls))
    long_r = routes[ridx]

    seed_candidates = top_contrib_nodes(long_r, dist, k=min(32, len(long_r)-2))
    seed = long_r[random.choice(seed_candidates)] if seed_candidates else random.choice(all_nodes)

    removed = [seed]
    while len(removed) < ruin_n:
        cand = [v for v in all_nodes if v not in removed]
        dmin, best_v = float("inf"), None
        for v in cand:
            for u in removed:
                d = dist[u, v]
                if d < dmin:
                    dmin, best_v = d, v
        removed.append(best_v)

    new_routes = [list(r) for r in routes]
    for v in removed:
        for ri in range(len(new_routes)):
            if v in new_routes[ri]:
                new_routes[ri].remove(v)
                break

    # Greedy reinsertion by minimal incremental cost (edges + node intrinsic length)
    for v in removed:
        cand_places = []
        for ri in range(len(new_routes)):
            r = new_routes[ri]
            for pos in range(1, len(r)):
                u, w = r[pos-1], r[pos]
                inc_edges = (dist[u, v] + dist[v, w]) - dist[u, w]
                inc_total = inc_edges + node_len[v]
                cand_places.append((inc_total, ri, pos))
        cand_places.sort(key=lambda x: x[0])
        _, ri, pos = cand_places[0]
        new_routes[ri].insert(pos, v)
    return new_routes

def reinforce_longest(routes: List[List[int]], dist: np.ndarray, node_len: np.ndarray, its=REINFORCE_2OPT_ITERS) -> List[List[int]]:
    """
    Intensification operator:
      - Apply intra-route improvements (2-opt + Or-opt) on the currently longest route.
    """
    ridx = int(np.argmax(lengths_all(routes, dist, node_len)))
    r = routes[ridx]
    r = two_opt_improve(r, dist, iters=its)
    for _ in range(REINFORCE_OROPT_ROUNDS):
        r2, g = or_opt_once(r, dist, ks=(1, 2, 3))
        if g <= 1e-12:
            break
        r = r2
    new_routes = [list(x) for x in routes]
    new_routes[ridx] = r
    return new_routes

# ======= Perturbations and equality checks =======
def routes_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    """
    Exact equality check for route sets (order-sensitive).
    """
    if len(a) != len(b):
        return False
    for r1, r2 in zip(a, b):
        if r1 != r2:
            return False
    return True

def targeted_kick(routes: List[List[int]], dist: np.ndarray, step_idx: int) -> List[List[int]]:
    """
    A lightweight perturbation (kick) operator to escape stagnation:
      - Remove a node from the route with the largest edge-sum,
      - Randomly insert it into another route.
    """
    new_routes = [list(r) for r in routes]
    Ls_edges = [sum(dist[r[i], r[i+1]] for i in range(len(r)-1)) for r in new_routes]
    ridx = int(np.argmax(Ls_edges))
    r = new_routes[ridx]
    if len(r) <= 3:
        return new_routes
    cands = top_contrib_nodes(r, dist, min(32, len(r)-2))
    if not cands:
        return new_routes
    i = random.choice(cands)
    node = r.pop(i)

    frac = min(1.0, step_idx / float(max(1, TOTAL_STEPS)))
    tries = max(6, int(16 * (1.0 - frac)))

    tgt_list = list(range(len(new_routes)))
    tgt_list.remove(ridx)
    random.shuffle(tgt_list)

    best_pos = None
    best_t = None
    best_cost = float("inf")
    for t in tgt_list:
        r2 = new_routes[t]
        for _ in range(tries):
            pos = random.randint(1, len(r2))
            cost = random.random()
            if cost < best_cost:
                best_cost, best_pos, best_t = cost, pos, t
    if best_pos is None:
        t = random.choice(tgt_list)
        pos = random.randint(1, len(new_routes[t]))
        new_routes[t].insert(pos, node)
    else:
        new_routes[best_t].insert(best_pos, node)
    return new_routes

def random_kick(routes: List[List[int]], dist: np.ndarray, step_idx: int) -> List[List[int]]:
    """
    Alias for targeted_kick (kept for extensibility).
    """
    return targeted_kick(routes, dist, step_idx)

# ========= GNN encoder =========
class SimpleGNN(nn.Module):
    """
    A minimal message-passing GNN that outputs:
      - graph-level embedding (mean over node embeddings),
      - node embeddings.
    """
    def __init__(self, in_dim, hid=64, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.layers.append(GNNLayer(last, hid))
            last = hid

    def forward(self, x, edge_index, num_nodes):
        h = x
        for gnn in self.layers:
            h = gnn(h, edge_index, num_nodes)
        graph_emb = h.mean(dim=0, keepdim=True)
        return graph_emb, h

class GNNLayer(nn.Module):
    """
    A simple neighborhood aggregation layer:
      h_i <- ReLU(W_self x_i + mean_{j in N(i)} W_nei x_j + b)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_nei  = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, edge_index, num_nodes):
        src, dst = edge_index
        self_part = self.lin_self(x)
        msg = self.lin_nei(x)
        agg = torch.zeros_like(msg)
        agg.index_add_(0, dst, msg[src])

        deg = torch.zeros(num_nodes, device=x.device, dtype=torch.float32)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg.clamp_min_(1.0).unsqueeze(1)

        nei_part = agg / deg
        out = self_part + nei_part + self.bias
        return torch.relu(out)

# ========= Policy and Value network =========
class PolicyValueNet(nn.Module):
    """
    Joint policy-value network with:
      - operator selection head,
      - parameter heads for inter-route operations,
      - value head,
      - auxiliary head predicting normalized improvements (dmax_norm, dstd_norm).
    """
    def __init__(self, node_feat_dim, gnn_hid, n_ops, stat_dim):
        super().__init__()
        self.gnn = SimpleGNN(node_feat_dim, hid=gnn_hid, layers=GNN_LAYERS)
        fused_dim = gnn_hid + stat_dim

        self.policy_op = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, n_ops)
        )

        # Parameterization heads for cross-route operators
        self.src_node_head = nn.Linear(gnn_hid, 1)     # node selection logits
        self.tgt_route_head = nn.Linear(gnn_hid, 1)    # target route selection logits
        self.pos_pair_proj = nn.Sequential(
            nn.Linear(gnn_hid * 2, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        self.value = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        # Auxiliary prediction head (self-supervised targets derived from realized improvements)
        self.aux = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # [dmax_norm, dstd_norm]
        )

    def forward(self, node_x, edge_index, num_nodes, stat_vec):
        graph_emb, node_h = self.gnn(node_x, edge_index, num_nodes)
        fused = torch.cat([graph_emb, stat_vec], dim=-1)
        op_logits = self.policy_op(fused)            # (1, n_ops)
        value  = self.value(fused).squeeze(-1)       # (1,)
        aux    = self.aux(fused).squeeze(0)          # (2,)
        return op_logits, value, aux, node_h

# ========= Feature engineering and graph construction =========
def build_knn_graph(coords: np.ndarray, k: int) -> np.ndarray:
    """
    Build a symmetric KNN graph using Euclidean distances.
    Returns edge_index with shape (2, E).
    """
    N = len(coords)
    dist = euclidian(coords)
    edges = []
    for i in range(N):
        order = np.argsort(dist[i])[1:k+1]
        for j in order:
            edges.append((i, int(j)))
            edges.append((int(j), i))
    return np.array(edges, dtype=np.int64).T

def routes_to_assign(routes: List[List[int]], N: int) -> np.ndarray:
    """
    Convert routes to node-to-route assignment vector (depot excluded).
    """
    assign = -np.ones(N, dtype=np.int64)
    for k, r in enumerate(routes):
        for v in r[1:-1]:
            assign[v] = k
    return assign

def routes_node_degree(routes: List[List[int]], N: int) -> np.ndarray:
    """
    Approximate node degree in the route graph (each internal node contributes 2).
    """
    deg = np.zeros(N, dtype=np.float32)
    for r in routes:
        for i in range(1, len(r)-1):
            deg[r[i]] += 2.0
    return deg

def build_node_features(coords: np.ndarray, routes: List[List[int]]) -> torch.Tensor:
    """
    Node features:
      [x, y, dist_to_depot, norm_x, norm_y, route_onehot(m), (deg/2)]
    The depot node feature is set to zeros.
    """
    N = len(coords)
    assign = routes_to_assign(routes, N)
    deg = routes_node_degree(routes, N)

    xy = coords.astype(np.float32)
    depot = xy[0]
    dist0 = np.linalg.norm(xy - depot[None, :], axis=1, keepdims=True)

    mean = xy[1:].mean(axis=0, keepdims=True)
    std = xy[1:].std(axis=0, keepdims=True) + 1e-6
    norm_xy = (xy - mean) / std

    route_oh = np.zeros((N, M_SALESMEN), dtype=np.float32)
    for i in range(N):
        if assign[i] >= 0:
            route_oh[i, assign[i]] = 1.0

    feat = np.concatenate(
        [xy, dist0, norm_xy, route_oh, (deg/2.0).reshape(-1, 1)],
        axis=1
    )
    feat[0] = 0.0
    return torch.tensor(feat, dtype=torch.float32, device=DEVICE)

def build_stat_vec(routes: List[List[int]], dist_all: np.ndarray, node_len: np.ndarray, t: int, best_soft: float) -> torch.Tensor:
    """
    Build global statistics features for the policy/value heads.
    """
    Ls = lengths_all(routes, dist_all, node_len)
    arr = np.array(Ls, dtype=np.float32)
    cur_max = float(arr.max())
    cur_mean = float(arr.mean())
    cur_std = float(arr.std() + 1e-9)
    soft = cur_max + LAMBDA_BAL * cur_std

    step_frac = float(min(t, TOTAL_STEPS)) / float(TOTAL_STEPS)
    vec = np.array([
        cur_max, cur_mean, cur_std, np.median(arr),
        float(t % 1000) / 1000.0, best_soft,
        cur_max / (cur_mean + 1e-6), soft / (best_soft + 1e-6),
        step_frac
    ], dtype=np.float32).reshape(1, -1)
    return torch.tensor(vec, dtype=torch.float32, device=DEVICE)

# ========= PPO rollout buffer =========
class PPOBuffer:
    """
    Storage for PPO rollouts.
    Stores route snapshots, actions, log-probabilities, values, rewards, and auxiliary targets.
    """
    def __init__(self):
        self.routes_snapshots = []
        self.actions = []          # operator index
        self.logprobs = []         # total log-prob (operator + parameters)
        self.op_logprobs = []
        self.param_logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.aux_targets = []      # [dmax_norm, dstd_norm]
        self.param_infos = []

    def add(self, routes, action, logprob, value, reward, done,
            aux_target=None, param_info=None, op_logp=None, param_logp=None):
        self.routes_snapshots.append([list(r) for r in routes])
        self.actions.append(int(action))
        self.logprobs.append(logprob.detach().view(1))
        self.values.append(value.detach().view(1))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.param_infos.append(param_info)
        self.op_logprobs.append(torch.tensor([0.0], device=DEVICE) if op_logp is None else op_logp.detach().view(1))
        self.param_logprobs.append(torch.tensor([0.0], device=DEVICE) if param_logp is None else param_logp.detach().view(1))
        if aux_target is None:
            self.aux_targets.append([0.0, 0.0])
        else:
            self.aux_targets.append([float(aux_target[0]), float(aux_target[1])])

    def clear(self):
        self.__init__()

# ========= PPO agent =========
class PPOAgent:
    """
    PPO agent that:
      - selects an operator (discrete action),
      - optionally selects parameters for cross-route operators,
      - updates policy/value networks with clipped surrogate objective,
      - includes an auxiliary regression loss on normalized improvements.
    """
    def __init__(self, node_feat_dim, stat_dim, n_ops):
        self.net = PolicyValueNet(node_feat_dim, GNN_HIDDEN, n_ops, stat_dim).to(DEVICE)
        self.opt = optim.Adam(self.net.parameters(), lr=PPO_LR)
        self.entropy_coef = PPO_ENTROPY
        self.aux_coef = 0.05  # weight of auxiliary regression loss

    @staticmethod
    def _mask_to_logits(mask_bool: torch.Tensor, fill: float = -1e9):
        """
        Convert a 0/1 mask to logits suitable for masking invalid categories.
        """
        return torch.where(
            mask_bool > 0,
            torch.zeros_like(mask_bool, dtype=torch.float32),
            torch.full_like(mask_bool, fill)
        )

    # ---- Parameter selection for inter-route relocate ----
    def _pick_param_inter_relocate(self, routes, node_h):
        """
        Parameterized sampling for inter-route relocate:
          - pick a source node from a (heuristically chosen) source route,
          - pick a target route,
          - pick an insertion position in the target route.
        """
        ridx = np.argmax([len(r) for r in routes])
        r_src = routes[ridx]
        idx_src_nodes = [i for i in range(1, len(r_src)-1)]
        if not idx_src_nodes:
            return None

        mask_src = torch.zeros(len(node_h), device=DEVICE)
        mask_src[torch.tensor([r_src[i] for i in idx_src_nodes], device=DEVICE)] = 1.0
        node_logits = self._mask_to_logits(mask_src).unsqueeze(1) + self.net.src_node_head(node_h)
        node_logits = node_logits.squeeze(1)
        dist_node = torch.distributions.Categorical(logits=node_logits)
        node_pick = dist_node.sample()
        logp_node = dist_node.log_prob(node_pick)

        if int(node_pick.item()) not in r_src:
            return None
        i_src = r_src.index(int(node_pick.item()))

        route_embs = []
        route_ids = []
        for rid, r in enumerate(routes):
            if rid == ridx or len(r) <= 2:
                continue
            nh = node_h[torch.tensor(r, device=DEVICE)]
            route_embs.append(nh.mean(dim=0, keepdim=True))
            route_ids.append(rid)
        if len(route_ids) == 0:
            return None

        route_embs = torch.cat(route_embs, dim=0)
        logits_tgt = self.net.tgt_route_head(route_embs).squeeze(1)
        dist_tgt = torch.distributions.Categorical(logits=logits_tgt)
        pick_idx = dist_tgt.sample()
        logp_tgt = dist_tgt.log_prob(pick_idx)
        ridx_tgt = route_ids[int(pick_idx.item())]
        r_tgt = routes[ridx_tgt]

        src_vec = node_h[node_pick]
        pos_logits = []
        pos_idx_list = list(range(1, len(r_tgt)))
        for pos in pos_idx_list:
            pre = r_tgt[pos-1]
            post = r_tgt[pos]
            pair = (node_h[pre] + node_h[post]) * 0.5
            feat = torch.cat([src_vec, pair], dim=-1)
            pos_logits.append(self.net.pos_pair_proj(feat))
        pos_logits = torch.cat(pos_logits, dim=0).squeeze(1)
        dist_pos = torch.distributions.Categorical(logits=pos_logits)
        pos_pick = dist_pos.sample()
        logp_pos = dist_pos.log_prob(pos_pick)
        pos_tgt = pos_idx_list[int(pos_pick.item())]

        logp_param = (logp_node + logp_tgt + logp_pos).view(1)
        return {
            "op": "inter_relocate",
            "r_src": ridx, "i_src": i_src,
            "r_tgt": ridx_tgt, "pos_tgt": pos_tgt,
            "logp_param": logp_param,
            "logp_parts": (logp_node.view(1), logp_tgt.view(1), logp_pos.view(1))
        }

    # ---- Parameter selection for inter-route swap ----
    def _pick_param_inter_swap(self, routes, node_h):
        """
        Parameterized sampling for inter-route swap:
          - pick node A from a source route,
          - pick a target route,
          - pick node B from the target route.
        """
        ridx_src = np.argmax([len(r) for r in routes])
        r_src = routes[ridx_src]
        idx_src_nodes = [i for i in range(1, len(r_src)-1)]
        if not idx_src_nodes:
            return None

        mask_src = torch.zeros(len(node_h), device=DEVICE)
        mask_src[torch.tensor([r_src[i] for i in idx_src_nodes], device=DEVICE)] = 1.0
        node_logits = self._mask_to_logits(mask_src).unsqueeze(1) + self.net.src_node_head(node_h)
        node_logits = node_logits.squeeze(1)
        dist_nodeA = torch.distributions.Categorical(logits=node_logits)
        pickA = dist_nodeA.sample()
        logp_A = dist_nodeA.log_prob(pickA)
        if int(pickA.item()) not in r_src:
            return None
        i_src = r_src.index(int(pickA.item()))

        tgt_ids = [rid for rid in range(len(routes)) if rid != ridx_src and len(routes[rid]) > 2]
        if not tgt_ids:
            return None

        route_embs = []
        for rid in tgt_ids:
            nh = node_h[torch.tensor(routes[rid], device=DEVICE)]
            route_embs.append(nh.mean(dim=0, keepdim=True))
        route_embs = torch.cat(route_embs, dim=0)
        logits_tgt = self.net.tgt_route_head(route_embs).squeeze(1)
        dist_tgt = torch.distributions.Categorical(logits=logits_tgt)
        pick_idx = dist_tgt.sample()
        logp_tgt = dist_tgt.log_prob(pick_idx)
        ridx_tgt = tgt_ids[int(pick_idx.item())]
        r_tgt = routes[ridx_tgt]

        idx_tgt_nodes = [i for i in range(1, len(r_tgt)-1)]
        mask_tgt = torch.zeros(len(node_h), device=DEVICE)
        mask_tgt[torch.tensor([r_tgt[i] for i in idx_tgt_nodes], device=DEVICE)] = 1.0
        node_logits_B = self._mask_to_logits(mask_tgt).unsqueeze(1) + self.net.src_node_head(node_h)
        node_logits_B = node_logits_B.squeeze(1)
        dist_nodeB = torch.distributions.Categorical(logits=node_logits_B)
        pickB = dist_nodeB.sample()
        logp_B = dist_nodeB.log_prob(pickB)
        if int(pickB.item()) not in r_tgt:
            return None
        j_tgt = r_tgt.index(int(pickB.item()))

        logp_param = (logp_A + logp_tgt + logp_B).view(1)
        return {
            "op": "inter_swap",
            "r_src": ridx_src, "i_src": i_src,
            "r_tgt": ridx_tgt, "j_tgt": j_tgt,
            "logp_param": logp_param,
            "logp_parts": (logp_A.view(1), logp_tgt.view(1), logp_B.view(1))
        }

    def select_action(self, node_x, edge_index, num_nodes, stat_vec, op_bias=None, routes=None):
        """
        Sample an operator action and (optionally) its parameters.
        Returns:
          action_idx, total_logp, value, param_dict, aux_pred, op_logp, param_logp
        """
        op_logits, value, aux, node_h = self.net(node_x, edge_index, num_nodes, stat_vec)
        if op_bias is not None:
            op_logits = op_logits + op_bias

        dist_op = torch.distributions.Categorical(logits=op_logits)
        a_op = dist_op.sample()
        logp_op = dist_op.log_prob(a_op)

        param = None
        param_logp = torch.tensor([0.0], device=DEVICE)

        if routes is not None:
            op_name = OPS[int(a_op.item())]
            if op_name == "inter_relocate":
                param = self._pick_param_inter_relocate(routes, node_h)
            elif op_name == "inter_swap":
                param = self._pick_param_inter_swap(routes, node_h)
            if param is not None and "logp_param" in param:
                param_logp = param["logp_param"]

        total_logp = (logp_op.view(1) + param_logp.view(1)).view(1)
        return (
            int(a_op.item()),
            total_logp.detach(),
            value.view(1).detach(),
            param,
            aux.detach(),
            logp_op.detach().view(1),
            param_logp.detach().view(1),
        )

    def _recompute_logps(self, node_x, edge_index, num_nodes, stat_vec, action_op: int, param_info, routes):
        """
        Recompute log-probabilities for PPO update (given stored actions/params) under the current policy.
        Also returns entropy and value estimate.
        """
        op_logits, value, aux, node_h = self.net(node_x, edge_index, num_nodes, stat_vec)
        dist_op = torch.distributions.Categorical(logits=op_logits)
        logp_op_new = dist_op.log_prob(torch.tensor(action_op, device=DEVICE))
        entropy_op = dist_op.entropy().mean()

        logp_param_new = torch.tensor(0.0, device=DEVICE)
        entropy_param = torch.tensor(0.0, device=DEVICE)

        if param_info is not None:
            # (The remainder of this function is unchanged; comments kept concise for readability.)
            # It reconstructs the factorized parameter distributions and evaluates stored decisions.
            if OPS[action_op] == "inter_relocate" and param_info.get("op") == "inter_relocate":
                rs, i, rt, pos = param_info["r_src"], param_info["i_src"], param_info["r_tgt"], param_info["pos_tgt"]

                idx_src_nodes = [k for k in range(1, len(routes[rs]) - 1)]
                if len(idx_src_nodes) > 0 and 0 < i < len(routes[rs]) - 1:
                    mask_src = torch.zeros(len(node_h), device=DEVICE)
                    mask_src[torch.tensor([routes[rs][k] for k in idx_src_nodes], device=DEVICE)] = 1.0
                    node_logits = self._mask_to_logits(mask_src).unsqueeze(1) + self.net.src_node_head(node_h)
                    node_logits = node_logits.squeeze(1)
                    dist_node = torch.distributions.Categorical(logits=node_logits)
                    node_id = routes[rs][i]
                    logp_node = dist_node.log_prob(torch.tensor(node_id, device=DEVICE))
                    entropy_param = entropy_param + dist_node.entropy().mean()
                else:
                    logp_node = torch.tensor(0.0, device=DEVICE)

                route_embs = []; route_ids = []
                for rid, r in enumerate(routes):
                    if rid == rs or len(r) <= 2:
                        continue
                    nh = node_h[torch.tensor(r, device=DEVICE)]
                    route_embs.append(nh.mean(dim=0, keepdim=True))
                    route_ids.append(rid)
                if len(route_ids) > 0 and rt in route_ids:
                    route_embs = torch.cat(route_embs, dim=0)
                    logits_tgt = self.net.tgt_route_head(route_embs).squeeze(1)
                    dist_tgt = torch.distributions.Categorical(logits=logits_tgt)
                    pick_idx = route_ids.index(rt)
                    logp_tgt = dist_tgt.log_prob(torch.tensor(pick_idx, device=DEVICE))
                    entropy_param = entropy_param + dist_tgt.entropy().mean()
                else:
                    logp_tgt = torch.tensor(0.0, device=DEVICE)

                if len(routes[rt]) >= 2 and 1 <= pos <= len(routes[rt]) - 1:
                    node_id = routes[rs][i]
                    src_vec = node_h[node_id]
                    pos_logits = []
                    pos_idx_list = list(range(1, len(routes[rt])))
                    for p in pos_idx_list:
                        pre = routes[rt][p-1]; post = routes[rt][p]
                        pair = (node_h[pre] + node_h[post]) * 0.5
                        feat = torch.cat([src_vec, pair], dim=-1)
                        pos_logits.append(self.net.pos_pair_proj(feat))
                    pos_logits = torch.cat(pos_logits, dim=0).squeeze(1)
                    dist_pos = torch.distributions.Categorical(logits=pos_logits)
                    pos_pick_idx = pos_idx_list.index(pos)
                    logp_pos = dist_pos.log_prob(torch.tensor(pos_pick_idx, device=DEVICE))
                    entropy_param = entropy_param + dist_pos.entropy().mean()
                else:
                    logp_pos = torch.tensor(0.0, device=DEVICE)

                logp_param_new = logp_node + logp_tgt + logp_pos

            elif OPS[action_op] == "inter_swap" and param_info.get("op") == "inter_swap":
                rs, i, rt, j = param_info["r_src"], param_info["i_src"], param_info["r_tgt"], param_info["j_tgt"]

                idx_src_nodes = [k for k in range(1, len(routes[rs]) - 1)]
                if len(idx_src_nodes) > 0 and 0 < i < len(routes[rs]) - 1:
                    mask_src = torch.zeros(len(node_h), device=DEVICE)
                    mask_src[torch.tensor([routes[rs][k] for k in idx_src_nodes], device=DEVICE)] = 1.0
                    node_logits = self._mask_to_logits(mask_src).unsqueeze(1) + self.net.src_node_head(node_h)
                    node_logits = node_logits.squeeze(1)
                    dist_nodeA = torch.distributions.Categorical(logits=node_logits)
                    node_idA = routes[rs][i]
                    logp_A = dist_nodeA.log_prob(torch.tensor(node_idA, device=DEVICE))
                    entropy_param = entropy_param + dist_nodeA.entropy().mean()
                else:
                    logp_A = torch.tensor(0.0, device=DEVICE)

                tgt_ids = [rid for rid in range(len(routes)) if rid != rs and len(routes[rid]) > 2]
                if len(tgt_ids) > 0 and rt in tgt_ids:
                    route_embs = []
                    for rid in tgt_ids:
                        nh = node_h[torch.tensor(routes[rid], device=DEVICE)]
                        route_embs.append(nh.mean(dim=0, keepdim=True))
                    route_embs = torch.cat(route_embs, dim=0)
                    logits_tgt = self.net.tgt_route_head(route_embs).squeeze(1)
                    dist_tgt = torch.distributions.Categorical(logits=logits_tgt)
                    pick_idx = tgt_ids.index(rt)
                    logp_tgt = dist_tgt.log_prob(torch.tensor(pick_idx, device=DEVICE))
                    entropy_param = entropy_param + dist_tgt.entropy().mean()
                else:
                    logp_tgt = torch.tensor(0.0, device=DEVICE)

                idx_tgt_nodes = [k for k in range(1, len(routes[rt]) - 1)]
                if len(idx_tgt_nodes) > 0 and 0 < j < len(routes[rt]) - 1:
                    mask_tgt = torch.zeros(len(node_h), device=DEVICE)
                    mask_tgt[torch.tensor([routes[rt][k] for k in idx_tgt_nodes], device=DEVICE)] = 1.0
                    node_logits_B = self._mask_to_logits(mask_tgt).unsqueeze(1) + self.net.src_node_head(node_h)
                    node_logits_B = node_logits_B.squeeze(1)
                    dist_nodeB = torch.distributions.Categorical(logits=node_logits_B)
                    node_idB = routes[rt][j]
                    logp_B = dist_nodeB.log_prob(torch.tensor(node_idB, device=DEVICE))
                    entropy_param = entropy_param + dist_nodeB.entropy().mean()
                else:
                    logp_B = torch.tensor(0.0, device=DEVICE)

                logp_param_new = logp_A + logp_tgt + logp_B

        return logp_op_new.view(1), logp_param_new.view(1), (entropy_op + entropy_param).view(1), value

    def ppo_update(self, buffer, coords, edge_index, dist_all, node_len, best_soft, update_steps, start_t, op_bias=None):
        """
        Perform PPO updates using collected rollouts.
        The policy is trained with:
          - clipped surrogate objective,
          - value regression,
          - entropy regularization,
          - auxiliary regression on normalized improvements.
        """
        N = len(coords)
        num_nodes = torch.tensor(N, device=DEVICE)

        node_x_list, stat_vec_list = [], []
        with torch.no_grad():
            for step, routes in enumerate(buffer.routes_snapshots):
                node_x = build_node_features(coords, routes)
                stat_vec = build_stat_vec(routes, dist_all, node_len, start_t + step, best_soft)
                node_x_list.append(node_x)
                stat_vec_list.append(stat_vec)

        old_total_logp = torch.cat(buffer.logprobs, dim=0).to(DEVICE).view(-1)
        old_values = torch.cat(buffer.values, dim=0).to(DEVICE).view(-1)
        actions = torch.tensor(buffer.actions, device=DEVICE, dtype=torch.long).view(-1)
        rewards = torch.tensor(buffer.rewards, device=DEVICE, dtype=torch.float32).view(-1)
        dones   = torch.tensor(buffer.dones,   device=DEVICE, dtype=torch.float32).view(-1)
        aux_tgt = torch.tensor(buffer.aux_targets, device=DEVICE, dtype=torch.float32)

        # Bootstrap value for the last state
        with torch.no_grad():
            last_node_x = build_node_features(coords, buffer.routes_snapshots[-1])
            last_stat   = build_stat_vec(buffer.routes_snapshots[-1], dist_all, node_len,
                                         start_t + len(buffer.routes_snapshots) - 1, best_soft)
            _, last_value, _, _ = self.net(last_node_x, edge_index, num_nodes, last_stat)
            last_value = last_value.view(1)
        values = torch.cat([old_values, last_value], dim=0)

        # GAE advantage estimation
        adv = torch.zeros_like(rewards, device=DEVICE)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + PPO_GAMMA * values[t+1] * mask - values[t]
            gae = delta + PPO_GAMMA * PPO_LAMBDA * mask * gae
            adv[t] = gae

        returns = adv + old_values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idxs = np.arange(len(rewards))
        actor_loss_out = 0.0

        for _epoch in range(PPO_EPOCHS):
            np.random.shuffle(idxs)
            early_stop = False
            for start in range(0, len(idxs), PPO_MINI_BATCH):
                mb_idx = idxs[start:start+PPO_MINI_BATCH]
                node_x_mb = [node_x_list[i] for i in mb_idx]
                stat_mb   = [stat_vec_list[i] for i in mb_idx]
                act_mb    = actions[mb_idx]
                old_total_logp_mb = old_total_logp[mb_idx]
                adv_mb    = adv[mb_idx]
                ret_mb    = returns[mb_idx]
                aux_mb    = aux_tgt[mb_idx]

                logp_new_list, entropy_list, values_new_list, aux_pred_list = [], [], [], []

                for i, (nx, st, a) in enumerate(zip(node_x_mb, stat_mb, act_mb.tolist())):
                    gidx = int(mb_idx[i])
                    param_info = buffer.param_infos[gidx]
                    op_logp_new, param_logp_new, ent_new, val_new = self._recompute_logps(
                        nx, edge_index, num_nodes, st, a, param_info, buffer.routes_snapshots[gidx]
                    )
                    logp_new_list.append((op_logp_new + param_logp_new).view(1))
                    entropy_list.append(ent_new.view(1))
                    values_new_list.append(val_new.view(1))

                    # Auxiliary prediction
                    _, _, aux_pred, _ = self.net(nx, edge_index, num_nodes, st)
                    aux_pred_list.append(aux_pred.view(1, 2))

                logp_new = torch.cat(logp_new_list, dim=0).view(-1)
                entropy  = torch.cat(entropy_list, dim=0).view(-1).mean()
                values_new = torch.cat(values_new_list, dim=0).view(-1)
                aux_pred   = torch.cat(aux_pred_list, dim=0)

                ratio = torch.exp(logp_new - old_total_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values_new - ret_mb)**2).mean()
                aux_loss   = ((aux_pred - aux_mb)**2).mean()

                loss = actor_loss + PPO_VF_COEF * value_loss - self.entropy_coef * entropy + self.aux_coef * aux_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), PPO_MAX_GRAD_NORM)
                self.opt.step()

                actor_loss_out = float(actor_loss.item())

                # KL-based early stopping and adaptive learning-rate/entropy control
                with torch.no_grad():
                    mb_kl = torch.clamp(old_total_logp_mb - logp_new, min=0).mean().item()
                if mb_kl > KL_TARGET * 2.0:
                    for g in self.opt.param_groups:
                        g["lr"] = max(1e-5, g["lr"] * LR_DECAY)
                    early_stop = True
                    break
                elif mb_kl < KL_TARGET * 0.5:
                    self.entropy_coef = float(np.clip(self.entropy_coef * 1.05, ENTROPY_MIN, ENTROPY_MAX))
                    for g in self.opt.param_groups:
                        g["lr"] = min(1e-3, g["lr"] * LR_GROW)
                else:
                    self.entropy_coef = float(np.clip(self.entropy_coef * 0.98, ENTROPY_MIN, ENTROPY_MAX))

            if early_stop:
                break

        return {"actor_loss": actor_loss_out}

# ========= Environment: operators with parameterization and fallback =========
def apply_operator(op: str, routes: List[List[int]], dist_all: np.ndarray,
                   coords: np.ndarray, node_len: np.ndarray, step_idx: int) -> List[List[int]]:
    """
    Apply a discrete operator. For parameterized operators (relocate/swap),
    this function uses heuristic versions. If the result is identical to the input,
    a kick perturbation is applied to avoid stagnation.
    """
    if op == "intra_2opt":
        Ls_edges = [sum(dist_all[r[i], r[i+1]] for i in range(len(r)-1)) for r in routes]
        ridx = int(np.argmax(Ls_edges))
        new_r = two_opt_improve(routes[ridx], dist_all, iters=70)
        new_routes = [list(r) for r in routes]
        new_routes[ridx] = new_r

    elif op == "intra_oropt":
        Ls_edges = [sum(dist_all[r[i], r[i+1]] for i in range(len(r)-1)) for r in routes]
        ridx = int(np.argmax(Ls_edges))
        r = routes[ridx]
        for _ in range(12):
            r2, g = or_opt_once(r, dist_all, ks=(1, 2, 3))
            if g <= 1e-12:
                break
            r = r2
        new_routes = [list(x) for x in routes]
        new_routes[ridx] = r

    elif op == "inter_relocate":
        new_routes = try_inter_relocate(routes, dist_all, node_len)

    elif op == "inter_swap":
        new_routes = try_inter_swap(routes, dist_all, node_len)

    elif op == "ruin_recreate":
        rr = ruin_recreate(routes, dist_all, coords, node_len, frac=SHAW_RUIN_FRAC)
        new_routes = reinforce_longest(rr, dist_all, node_len, its=REINFORCE_2OPT_ITERS//2)

    elif op == "reinforce_longest":
        new_routes = reinforce_longest(routes, dist_all, node_len, its=REINFORCE_2OPT_ITERS)

    else:
        new_routes = [list(r) for r in routes]

    # Fallback perturbation if the operator produces no change
    if routes_equal(new_routes, routes):
        new_routes = random_kick(routes, dist_all, step_idx)
    return new_routes

def apply_operator_with_param(op, routes, dist_all, coords, node_len, step_idx, param) -> Optional[List[List[int]]]:
    """
    Apply a parameterized inter-route operator using the sampled parameters.
    Returns None if parameters are invalid or the operation is infeasible.
    """
    if param is None:
        return None
    try:
        new_routes = [list(r) for r in routes]

        if op == "inter_relocate" and param.get("op") == "inter_relocate":
            rs, i, rt, pos = param["r_src"], param["i_src"], param["r_tgt"], param["pos_tgt"]
            if rs == rt:
                return None
            if not (0 < i < len(new_routes[rs]) - 1):
                return None
            node = new_routes[rs].pop(i)
            pos = max(1, min(pos, len(new_routes[rt])))
            new_routes[rt].insert(pos, node)
            return new_routes

        if op == "inter_swap" and param.get("op") == "inter_swap":
            rs, i, rt, j = param["r_src"], param["i_src"], param["r_tgt"], param["j_tgt"]
            if rs == rt:
                return None
            if not (0 < i < len(new_routes[rs]) - 1):
                return None
            if not (0 < j < len(new_routes[rt]) - 1):
                return None
            new_routes[rs][i], new_routes[rt][j] = new_routes[rt][j], new_routes[rs][i]
            return new_routes

        return None
    except Exception:
        return None

# ========= Serialization and logging helpers =========
def save_best_routes_json(best_routes: List[List[int]], dist_all: np.ndarray, path: str,
                          coordidx_to_orig: Dict[int, int], node_len: np.ndarray):
    """
    Save best routes to JSON with original task IDs and route lengths.
    """
    obj = []
    for k, r in enumerate(best_routes):
        mapped_nodes = [int(coordidx_to_orig.get(v, v)) for v in r]
        obj.append({
            "route_id": k,
            "nodes": mapped_nodes,
            "length": tour_length(r, dist_all, node_len)
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_progress_csv(t: int, routes: List[List[int]], dist_all: np.ndarray,
                        node_len: np.ndarray, cur_max: float, cur_soft: float, path: str):
    """
    Append per-step route statistics to a CSV file.
    """
    Ls = lengths_all(routes, dist_all, node_len)
    row = {"step": t, "cur_max": cur_max, "cur_soft": cur_soft}
    for i, L in enumerate(Ls):
        row[f"route{i}_len"] = L
        row[f"route{i}_nodes"] = len(routes[i]) - 1
    hdr = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=hdr)

def append_losses_csv(t: int, stats: dict, path: str):
    """
    Append PPO training losses to a CSV file.
    """
    row = {"step": t, "actor_loss": stats.get("actor_loss", 0.0)}
    hdr = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=hdr)

def draw_loss_curves(loss_csv: str, out_png: str):
    """
    Plot and save PPO loss curves from the logged CSV file.
    """
    if not os.path.exists(loss_csv):
        return
    df = pd.read_csv(loss_csv)
    if df.empty:
        return

    def smooth(x, k=5):
        if len(x) < k:
            return x
        return pd.Series(x).rolling(k, min_periods=1).mean().values

    plt.figure(figsize=(7, 4.2))
    if "actor_loss" in df.columns:
        plt.plot(df["step"], smooth(df["actor_loss"]), label="actor_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("PPO Actor Loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)

# ========= Main entry =========
def main():
    # ---- Load task coordinates ----
    df = pd.read_csv(CSV_PATH)
    coords_list = []
    for s in df["coordinates_mean"]:
        x, y = s.strip("[]").split(",")
        coords_list.append([float(x), float(y)])
    tasks = np.array(coords_list, dtype=np.float64)

    # Original task IDs (as provided in the first column of the coordinate CSV)
    orig_ids = df.iloc[:, 0].values.tolist()
    try:
        orig_ids = [int(v) for v in orig_ids]
    except Exception:
        orig_ids = [str(v) for v in orig_ids]

    # Internal index → original task ID mapping (internal 0 is depot)
    coordidx_to_orig = {0: 0}
    for i, oid in enumerate(orig_ids, start=1):
        coordidx_to_orig[i] = oid

    # ---- Load node intrinsic lengths ----
    node_len_map: Dict = {}
    if os.path.exists(ROADLEN_CSV):
        df_len = pd.read_csv(ROADLEN_CSV, header=None, names=["id", "length"])
        for _, row in df_len.iterrows():
            key_raw = row["id"]
            try:
                key_int = int(key_raw)
                node_len_map[key_int] = float(row["length"])
            except Exception:
                node_len_map[str(key_raw)] = float(row["length"])
    else:
        print("[WARN] road_length.csv not found; all node intrinsic lengths are treated as 0.")

    # Depot is fixed at (0, 0)
    depot = np.array([[0.0, 0.0]], dtype=np.float64)
    coords = np.concatenate([depot, tasks], axis=0)
    N = len(coords)

    # Construct per-node intrinsic length vector aligned with internal indices
    node_len = np.zeros(N, dtype=np.float64)
    for idx in range(1, N):
        oid = coordidx_to_orig[idx]
        ln = None
        if isinstance(oid, int) and oid in node_len_map:
            ln = node_len_map[oid]
        elif str(oid) in node_len_map:
            ln = node_len_map[str(oid)]
        node_len[idx] = float(ln) if ln is not None else 0.0

    print(f"Loaded {len(tasks)} tasks. Depot fixed at (0, 0). Device={DEVICE.type}")
    print(f"Node length stats: min={node_len[1:].min():.4f}, max={node_len[1:].max():.4f}, mean={node_len[1:].mean():.4f}")

    # ---- Build hybrid distance matrix ----
    dist_all = build_hybrid_distance(coords, orig_ids, DISTM_PATH)

    # ---- Build GNN graph (Euclidean KNN) ----
    edge_index_np = build_knn_graph(coords, KNN_K)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=DEVICE)

    # ---- Initialize solution ----
    t0 = time.time()
    routes = build_init(coords, M_SALESMEN, dist_all, node_len)
    cur_max, cur_soft = compute_obj(routes, dist_all, node_len)
    best_routes = [list(r) for r in routes]
    best_max, best_soft = cur_max, cur_soft
    best_max_alone = cur_max

    print(f"Init built in {time.time()-t0:.2f}s | init max={cur_max:.2f} soft={cur_soft:.2f}")
    for k, r in enumerate(routes):
        print(f"  Route[{k}] nodes={len(r)-1} len={tour_length(r, dist_all, node_len):.2f}")

    # ---- Persist initial artifacts ----
    append_progress_csv(0, routes, dist_all, node_len, cur_max, cur_soft, PROGRESS_CSV)
    save_best_routes_json(best_routes, dist_all, BEST_JSON, coordidx_to_orig, node_len)

    # ---- PPO initialization ----
    node_feat_dim = 5 + M_SALESMEN + 1  # [x,y,dist0,normx,normy] + route one-hot + deg/2
    stat_dim = 9
    agent = PPOAgent(node_feat_dim, stat_dim, N_OPS)
    buffer = PPOBuffer()

    ema_reward = 0.0
    op_credit = np.zeros(N_OPS, dtype=np.float32)

    start = time.time()
    for t in range(1, TOTAL_STEPS + 1):
        step_frac = float(min(t, TOTAL_STEPS)) / float(TOTAL_STEPS)
        accept_tau = ACCEPT_TAU_START + (ACCEPT_TAU_END - ACCEPT_TAU_START) * step_frac
        op_credit_beta = OP_CREDIT_BETA_START + (OP_CREDIT_BETA_END - OP_CREDIT_BETA_START) * step_frac

        node_x = build_node_features(coords, routes)
        stat_vec = build_stat_vec(routes, dist_all, node_len, t, best_soft)

        # Operator bias based on historical success (credit)
        op_bias = torch.tensor(op_credit_beta * op_credit, dtype=torch.float32, device=DEVICE).view(1, -1)

        # Action selection (round-robin warmup followed by ε-greedy + policy sampling)
        if t <= RR_EARLY_STEPS:
            action_idx = (t - 1) % len(OPS)
            op = OPS[action_idx]
            logp_total = torch.tensor([0.0], device=DEVICE)
            value = torch.tensor([0.0], device=DEVICE)
            param = None
            op_logp = torch.tensor([0.0], device=DEVICE)
            param_logp = torch.tensor([0.0], device=DEVICE)
        else:
            eps = max(EPS_MIN, EPS0 * (1.0 - (t - 1 - RR_EARLY_STEPS) / max(1, EPS_DECAY_STEPS)))
            if random.random() < eps:
                action_idx = random.randrange(len(OPS))
                op = OPS[action_idx]
                logp_total = torch.tensor([0.0], device=DEVICE)
                value = torch.tensor([0.0], device=DEVICE)
                param = None
                op_logp = torch.tensor([0.0], device=DEVICE)
                param_logp = torch.tensor([0.0], device=DEVICE)
            else:
                action_idx, logp_total, value, param, aux_pred, op_logp, param_logp = agent.select_action(
                    node_x, edge_index, torch.tensor(N, device=DEVICE), stat_vec, op_bias=op_bias, routes=routes
                )
                op = OPS[action_idx]

        # Pre-move statistics (using total lengths)
        old_Ls = lengths_all(routes, dist_all, node_len)
        old_max = float(np.max(old_Ls))
        old_std = float(np.std(old_Ls))

        # Apply operator (parameterized execution preferred; fallback to heuristic if invalid)
        routes2 = None
        if op in ("inter_relocate", "inter_swap"):
            routes2 = apply_operator_with_param(op, routes, dist_all, coords, node_len, t, param)
        if routes2 is None:
            routes2 = apply_operator(op, routes, dist_all, coords, node_len, t)

        # Post-move statistics
        new_Ls = lengths_all(routes2, dist_all, node_len)
        new_max = float(np.max(new_Ls))
        new_std = float(np.std(new_Ls))

        # Reject extremely deteriorating moves unless std improves substantially
        worse_enough = (new_max > old_max * (1.0 + accept_tau))
        std_improve  = (new_std < old_std * (1.0 - 0.5 * accept_tau))
        if worse_enough and (not std_improve):
            routes2 = routes
            new_Ls  = old_Ls
            new_max = old_max
            new_std = old_std

        # Lexicographic reward shaping based on (max, std)
        if routes_equal(routes2, routes):
            inst_reward = SMALL_NEG_REWARD
            success = 0.0
        else:
            dmax = (old_max - new_max)
            dstd = (old_std - new_std)
            primary_thr = DELTA_MAX_PRIMARY_THR_FRAC * max(1.0, old_max)
            if abs(dmax) > primary_thr:
                if dmax > 0 and dstd < 0:
                    dstd = max(dstd, -STD_WORSEN_MARGIN * dmax)
                max_cap = REWARD_CLIP_FRAC * max(1.0, old_max)
                std_cap = STD_CLIP_FRAC * max(1.0, old_std)
                dmax_c = float(np.clip(dmax, -max_cap, max_cap))
                dstd_c = float(np.clip(dstd, -std_cap, std_cap))
                inst_reward = (dmax_c + 0.25 * dstd_c) / (max(1.0, old_max))
            else:
                std_cap = STD_CLIP_FRAC * max(1.0, old_std)
                dstd_c = float(np.clip(dstd, -std_cap, std_cap))
                inst_reward = 0.5 * dstd_c / (max(1.0, old_std))

            success = 1.0 if dmax > 0 else (0.5 if abs(dmax) <= primary_thr and dstd > 0 else 0.0)

        # Exponential moving average reward for stability
        ema_reward = EMA_BETA * ema_reward + (1.0 - EMA_BETA) * inst_reward
        reward = ema_reward

        # Update operator credit (success statistics)
        op_credit[action_idx] = OP_CREDIT_DECAY * op_credit[action_idx] + (1.0 - OP_CREDIT_DECAY) * success

        # Auxiliary targets (normalized improvements)
        dmax_norm = ((old_max - new_max) / max(1.0, old_max))
        dstd_norm = ((old_std - new_std) / max(1.0, old_std))

        # State transition
        routes = routes2
        cur_max = new_max
        cur_soft = float(new_max + LAMBDA_BAL * new_std)

        # Update best solution (soft criterion)
        updated_best = False
        if cur_soft < best_soft - 1e-12:
            best_soft, best_max = cur_soft, cur_max
            best_routes = [list(r) for r in routes]
            updated_best = True

        # Track best by primary objective alone
        if cur_max < best_max_alone - 1e-12:
            best_max_alone = cur_max

        # Periodic intensification
        if t % INTENSIFY_EVERY == 0:
            routes = reinforce_longest(routes, dist_all, node_len, its=REINFORCE_2OPT_ITERS)
            cur_max, cur_soft = compute_obj(routes, dist_all, node_len)

        # Periodic diversification (restart from best + light perturbation)
        if t % DIVERSIFY_EVERY == 0 and t > RR_EARLY_STEPS:
            routes = [list(r) for r in best_routes]
            routes = ruin_recreate(routes, dist_all, coords, node_len, frac=LIGHT_PERTURB_FRAC)
            routes = reinforce_longest(routes, dist_all, node_len, its=REINFORCE_2OPT_ITERS//2)
            cur_max, cur_soft = compute_obj(routes, dist_all, node_len)

        # Logging
        if (t % PRINT_EVERY == 0) or (t == 1) or updated_best:
            append_progress_csv(t, routes, dist_all, node_len, cur_max, cur_soft, PROGRESS_CSV)
        if updated_best:
            save_best_routes_json(best_routes, dist_all, BEST_JSON, coordidx_to_orig, node_len)

        # Store transition in PPO buffer
        buffer.add(
            routes, action_idx, logp_total, value, reward, 0.0,
            aux_target=(dmax_norm, dstd_norm), param_info=param,
            op_logp=op_logp, param_logp=param_logp
        )

        # PPO update
        if t % PPO_BATCH == 0 and t > RR_EARLY_STEPS:
            stats = agent.ppo_update(
                buffer, coords, edge_index, dist_all, node_len,
                best_soft, update_steps=PPO_EPOCHS, start_t=t - PPO_BATCH + 1,
                op_bias=torch.tensor(op_credit_beta * op_credit, dtype=torch.float32, device=DEVICE).view(1, -1)
            )
            append_losses_csv(t, stats, PPO_LOSS_CSV)
            buffer.clear()

        # Console progress
        if t % PRINT_EVERY == 0 or t == 1:
            Ls = lengths_all(routes, dist_all, node_len)
            ridx = int(np.argmax(Ls))
            print(
                f"[{t:5d}/{TOTAL_STEPS}] op={OPS[action_idx]:18s} reward(inst)={inst_reward:+8.4f} "
                f"cur_max={cur_max:9.2f} cur_soft={cur_soft:9.2f} "
                f"best_max={best_max:9.2f} best_soft={best_soft:9.2f} "
                f"best_max_alone={best_max_alone:9.2f} time={time.time()-start:6.1f}s"
            )
            for k, Lk in enumerate(Ls):
                prefix = " *" if k == ridx else "  "
                print(f"{prefix} Route[{k}] len={Lk:.2f} nodes={len(routes[k])-1}")

    print("\n=== Done ===")
    print(f"Best by soft -> max: {best_max:.4f} | soft: {best_soft:.4f}")
    print(f"Best max alone: {best_max_alone:.4f}")

    # Plot training curves
    draw_loss_curves(PPO_LOSS_CSV, PPO_LOSS_PNG)
    print("\nSaved files:")
    print(f"  - {BEST_JSON}")
    print(f"  - {PROGRESS_CSV}")
    print(f"  - {PPO_LOSS_CSV}")
    print(f"  - {PPO_LOSS_PNG}")

if __name__ == "__main__":
    main()


