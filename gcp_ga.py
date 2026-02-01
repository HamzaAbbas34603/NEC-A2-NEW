from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np


# ----------------------------
# Graph loading (.col format)
# -----------------------------
def load_col_graph(path: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Load a graph from DIMACS-like .col files:
    - 'p edge V E' line defines number of vertices V
    - 'e u v' lines define edges (1-indexed usually)
    Returns:
      V (int), edges (list of (u, v) 0-indexed)
    """
    V = None
    edges: List[Tuple[int, int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if parts[0] == "p":
                # e.g. p edge 500 12500
                V = int(parts[2])
            elif parts[0] == "e":
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                if u != v:
                    edges.append((u, v))
    if V is None:
        raise ValueError(f"Could not find 'p edge V E' line in {path}")
    return V, edges


def build_adjacency(V: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(V)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


# -----------------------------
# GA definitions
# -----------------------------
@dataclass
class GAConfig:
    # Core
    population_size: int
    generations: int
    elite_count: int = 1

    # Operators
    selection: str = "tournament"  # tournament | roulette
    crossover: str = "one_point"   # one_point | uniform
    mutation: str = "random_reset" # random_reset | greedy

    # Rates & params
    crossover_rate: float = 0.9
    mutation_rate: float = 0.05
    tournament_k: int = 3

    # Fitness penalty
    conflict_weight: float = 50.0  # alpha in: alpha*conflicts + colors_used

    # Stationary detection
    patience: int = 100  # stop if no improvement for this many generations

    # Colors search space
    # If None, defaults to V (safe upper bound).
    max_colors: Optional[int] = None

    seed: Optional[int] = 42


@dataclass
class GAResult:
    best_chromosome: np.ndarray
    best_fitness: float
    best_conflicts: int
    best_colors_used: int
    fitness_history: List[float]
    time_seconds: float
    generations_ran: int
    config: GAConfig


# -----------------------------
# Fitness / evaluation
# -----------------------------
def count_conflicts(chrom: np.ndarray, edges: List[Tuple[int, int]]) -> int:
    # Count conflicting edges (u,v) where same color
    # edges are undirected; each listed once in input
    conflicts = 0
    for u, v in edges:
        if chrom[u] == chrom[v]:
            conflicts += 1
    return conflicts


def colors_used(chrom: np.ndarray) -> int:
    return int(len(np.unique(chrom)))


def fitness(chrom: np.ndarray, edges: List[Tuple[int, int]], conflict_weight: float) -> float:
    c = count_conflicts(chrom, edges)
    k = colors_used(chrom)
    return conflict_weight * c + k


# -----------------------------
# Initialization
# -----------------------------
def init_population(pop_size: int, V: int, K: int, rng: random.Random) -> np.ndarray:
    # shape: (pop_size, V)
    return np.array([[rng.randrange(K) for _ in range(V)] for _ in range(pop_size)], dtype=np.int32)


# -----------------------------
# Selection methods (2 required)
# -----------------------------
def select_parent_tournament(pop: np.ndarray, fit: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    idxs = [rng.randrange(len(pop)) for _ in range(k)]
    best_i = min(idxs, key=lambda i: fit[i])
    return pop[best_i].copy()


def select_parent_roulette(pop: np.ndarray, fit: np.ndarray, rng: random.Random) -> np.ndarray:
    # Roulette on inverse fitness (lower is better)
    # Avoid division by zero:
    inv = 1.0 / (fit + 1e-9)
    probs = inv / inv.sum()
    choice = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if choice <= cum:
            return pop[i].copy()
    return pop[-1].copy()


# -----------------------------
# Crossover methods (2 required)
# -----------------------------
def crossover_one_point(p1: np.ndarray, p2: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    V = len(p1)
    if V < 2:
        return p1.copy(), p2.copy()
    cut = rng.randrange(1, V)
    c1 = np.concatenate([p1[:cut], p2[cut:]]).astype(np.int32)
    c2 = np.concatenate([p2[:cut], p1[cut:]]).astype(np.int32)
    return c1, c2


def crossover_uniform(p1: np.ndarray, p2: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    V = len(p1)
    mask = np.array([rng.random() < 0.5 for _ in range(V)], dtype=bool)
    c1 = p1.copy()
    c2 = p2.copy()
    c1[mask] = p2[mask]
    c2[mask] = p1[mask]
    return c1.astype(np.int32), c2.astype(np.int32)


# -----------------------------
# Mutation methods (2 required)
# -----------------------------
def mutate_random_reset(chrom: np.ndarray, K: int, mutation_rate: float, rng: random.Random) -> np.ndarray:
    V = len(chrom)
    out = chrom.copy()
    for i in range(V):
        if rng.random() < mutation_rate:
            out[i] = rng.randrange(K)
    return out


def mutate_greedy(chrom: np.ndarray, adj: List[List[int]], K: int, mutation_rate: float, rng: random.Random) -> np.ndarray:
    """
    Greedy mutation:
    - With prob mutation_rate per gene, if gene is in conflict, assign least-conflicting color.
    - Otherwise do nothing.
    """
    V = len(chrom)
    out = chrom.copy()

    for v in range(V):
        if rng.random() >= mutation_rate:
            continue

        # check if v is currently conflicting
        cv = out[v]
        conflicted = any(out[n] == cv for n in adj[v])
        if not conflicted:
            continue

        # try all colors, pick one minimizing local conflicts
        best_color = cv
        best_local = 10**9
        for c in range(K):
            local = 0
            for n in adj[v]:
                if out[n] == c:
                    local += 1
            if local < best_local:
                best_local = local
                best_color = c

        out[v] = best_color

    return out


# -----------------------------
# GA main loop
# -----------------------------
def run_ga(V: int, edges: List[Tuple[int, int]], cfg: GAConfig) -> GAResult:
    if cfg.seed is not None:
        rng = random.Random(cfg.seed)
        np.random.seed(cfg.seed)
    else:
        rng = random.Random()

    adj = build_adjacency(V, edges)
    K = cfg.max_colors if cfg.max_colors is not None else V

    pop = init_population(cfg.population_size, V, K, rng)

    fit = np.array([fitness(ind, edges, cfg.conflict_weight) for ind in pop], dtype=np.float64)

    best_idx = int(np.argmin(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])
    best_conf = count_conflicts(best, edges)
    best_k = colors_used(best)

    history = [best_fit]
    no_improve = 0
    start = time.time()

    for gen in range(cfg.generations):
        # Elitism
        elite_idxs = np.argsort(fit)[: cfg.elite_count]
        new_pop = [pop[i].copy() for i in elite_idxs]

        # Reproduction
        while len(new_pop) < cfg.population_size:
            # selection
            if cfg.selection == "tournament":
                p1 = select_parent_tournament(pop, fit, cfg.tournament_k, rng)
                p2 = select_parent_tournament(pop, fit, cfg.tournament_k, rng)
            elif cfg.selection == "roulette":
                p1 = select_parent_roulette(pop, fit, rng)
                p2 = select_parent_roulette(pop, fit, rng)
            else:
                raise ValueError(f"Unknown selection: {cfg.selection}")

            # crossover
            if rng.random() < cfg.crossover_rate:
                if cfg.crossover == "one_point":
                    c1, c2 = crossover_one_point(p1, p2, rng)
                elif cfg.crossover == "uniform":
                    c1, c2 = crossover_uniform(p1, p2, rng)
                else:
                    raise ValueError(f"Unknown crossover: {cfg.crossover}")
            else:
                c1, c2 = p1.copy(), p2.copy()

            # mutation
            if cfg.mutation == "random_reset":
                c1 = mutate_random_reset(c1, K, cfg.mutation_rate, rng)
                c2 = mutate_random_reset(c2, K, cfg.mutation_rate, rng)
            elif cfg.mutation == "greedy":
                c1 = mutate_greedy(c1, adj, K, cfg.mutation_rate, rng)
                c2 = mutate_greedy(c2, adj, K, cfg.mutation_rate, rng)
            else:
                raise ValueError(f"Unknown mutation: {cfg.mutation}")

            new_pop.append(c1)
            if len(new_pop) < cfg.population_size:
                new_pop.append(c2)

        pop = np.array(new_pop, dtype=np.int32)
        fit = np.array([fitness(ind, edges, cfg.conflict_weight) for ind in pop], dtype=np.float64)

        gen_best_idx = int(np.argmin(fit))
        gen_best = pop[gen_best_idx].copy()
        gen_best_fit = float(fit[gen_best_idx])

        if gen_best_fit + 1e-12 < best_fit:
            best_fit = gen_best_fit
            best = gen_best.copy()
            best_conf = count_conflicts(best, edges)
            best_k = colors_used(best)
            no_improve = 0
        else:
            no_improve += 1

        history.append(best_fit)

        if no_improve >= cfg.patience:
            # stationary state detected
            end = time.time()
            return GAResult(
                best_chromosome=best,
                best_fitness=best_fit,
                best_conflicts=best_conf,
                best_colors_used=best_k,
                fitness_history=history,
                time_seconds=end - start,
                generations_ran=gen + 1,
                config=cfg,
            )

    end = time.time()
    return GAResult(
        best_chromosome=best,
        best_fitness=best_fit,
        best_conflicts=best_conf,
        best_colors_used=best_k,
        fitness_history=history,
        time_seconds=end - start,
        generations_ran=cfg.generations,
        config=cfg,
    )


def pretty_solution_summary(res: GAResult) -> str:
    cfg = res.config
    return (
        f"Best fitness={res.best_fitness:.3f} | conflicts={res.best_conflicts} | "
        f"colors_used={res.best_colors_used} | gens={res.generations_ran} | "
        f"time={res.time_seconds:.2f}s | sel={cfg.selection} | cross={cfg.crossover} | mut={cfg.mutation} | "
        f"pop={cfg.population_size} | cr={cfg.crossover_rate} | mr={cfg.mutation_rate} | alpha={cfg.conflict_weight}"
    )
