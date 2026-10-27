"""OMCDE: Opposition-based Multi-Stage Matrix-Based Competitive DE.

A NumPy implementation of the algorithm described by Sarhangzadeh,
Rahnamayan, and Oner. Optimization internals are vectorized over candidates.

Implementation decisions absent or inconsistent in the paper are documented
in the repository README and notebook.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional, Union

import numpy as np

Array = np.ndarray
BatchObjective = Callable[[Array], Array]


@dataclass
class OMCDEResult:
    """Optimization outcome and history (minimization)."""

    x: Array
    fun: float
    nfev: int
    nit: int
    elapsed_seconds: float
    history: dict
    final_population: Array
    final_fitness: Array

    def __repr__(self) -> str:
        return (
            f"OMCDEResult(fun={self.fun:.6g}, nfev={self.nfev}, "
            f"nit={self.nit}, elapsed_seconds={self.elapsed_seconds:.3f})"
        )


def _evaluate(objective: BatchObjective, candidates: Array) -> Array:
    """Evaluate a full batch; input (N, D) -> output (N,)."""
    values = np.asarray(objective(candidates), dtype=float).reshape(-1)
    if values.size != len(candidates):
        raise ValueError(
            "objective must map an (N, D) array to N fitness values; "
            f"received shape {values.shape} for N={len(candidates)}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("objective returned NaN or infinite fitness values")
    return values


def _controls(
    rng: np.random.Generator, memory_f: Array, memory_cr: Array, size: int
) -> tuple[Array, Array]:
    """Sample per-individual SHADE-style F and CR vectors (Eq. 15)."""
    memory_indices = rng.integers(0, len(memory_f), size=size)
    f = memory_f[memory_indices] + 0.1 * rng.standard_cauchy(size)
    # Reject nonpositive Cauchy samples in batches; no candidate-by-candidate loop.
    invalid = f <= 0.0
    while np.any(invalid):
        f[invalid] = memory_f[memory_indices[invalid]] + 0.1 * rng.standard_cauchy(
            np.count_nonzero(invalid)
        )
        invalid = f <= 0.0
    f = np.minimum(f, 1.0)
    cr = np.clip(
        rng.normal(memory_cr[memory_indices], 0.1, size=size), 0.0, 1.0
    )
    return f, cr


def _weighted_lehmer(values: Array, improvements: Array) -> Optional[float]:
    """Improvement-weighted Lehmer mean (Eq. 16)."""
    denominator = np.dot(improvements, values)
    if denominator <= 0:
        return None
    return float(np.dot(improvements, values * values) / denominator)


def _random_differences(
    rng: np.random.Generator, population: Array, archive: Array, targets: Array
) -> Array:
    """Batch sample r1 != r2, both different from each target.

    Indices refer to the union of current population and external archive.
    Rank-skipping sampling avoids a per-individual rejection loop.
    """
    pool = np.concatenate((population, archive), axis=0)
    size = len(pool)
    # Population contains at least four members, so this is always possible.
    r1 = rng.integers(0, size - 1, size=targets.size)
    r1 += r1 >= targets
    low = np.minimum(r1, targets)
    high = np.maximum(r1, targets)
    r2 = rng.integers(0, size - 2, size=targets.size)
    r2 += r2 >= low
    r2 += r2 >= high
    return pool[r1] - pool[r2]


def _competitive_mutants(
    rng: np.random.Generator,
    population: Array,
    fitness: Array,
    factors: Array,
    archive: Array,
    global_best: Array,
    greedy: bool,
) -> Array:
    """Vectorized triplet competition and two-stage mutation (Eqs. 2-4, 12-14).

    When N is not divisible by 3, pad the final triplet with *companions*
    borrowed from other groups. Padded entries help rank candidates but do
    not receive another trial; each original candidate gets exactly one trial.
    """
    n, dim = population.shape
    shuffled = rng.permutation(n)
    remainder = n % 3
    padded = 0 if remainder == 0 else 3 - remainder
    if padded:
        full = n - remainder
        companions = rng.choice(shuffled[:full], size=padded, replace=False)
        shuffled = np.concatenate((shuffled, companions))

    eligible = np.concatenate((np.ones(n, bool), np.zeros(padded, bool)))
    groups = shuffled.reshape(-1, 3)
    eligible = eligible.reshape(-1, 3)
    # Each triplet is ranked by ascending fitness (best, medium, worst).
    order = np.argsort(fitness[groups], axis=1, kind="stable")
    ranked_ids = np.take_along_axis(groups, order, axis=1)
    eligible = np.take_along_axis(eligible, order, axis=1)

    individuals = population[ranked_ids]
    best, middle, worst = individuals[:, 0], individuals[:, 1], individuals[:, 2]
    f = factors[ranked_ids, None]  # (triplets, 3, 1)
    fb, fm, fw = f[:, 0], f[:, 1], f[:, 2]
    differences = _random_differences(
        rng, population, archive, ranked_ids.ravel()
    ).reshape(-1, 3, dim)

    if greedy:
        # Equations 12-14, with the global best broadcast over all triplets.
        vb = best + fb * (global_best - best) + fb * differences[:, 0]
        vm = (
            middle
            + fm * (global_best - middle)
            + fm * (best - middle)
            + fm * differences[:, 1]
        )
        vw = (
            worst
            + fw * (global_best - worst)
            + fw * (best - worst)
            + fw * (middle - worst)
        )
    else:
        # Equations 2-4: original TCDE-style competitive behavior.
        vb = best + fb * differences[:, 0]
        vm = middle + fm * (best - middle) + fm * differences[:, 1]
        vw = worst + fw * (best - worst) + fw * (middle - worst)

    mutated = np.stack((vb, vm, vw), axis=1)
    result = np.empty_like(population)
    result[ranked_ids[eligible]] = mutated[eligible]
    return result


def _repair_bounds(
    trial: Array, parent: Array, lower: Array, upper: Array, method: str
) -> Array:
    """Bound handling; midpoint is a documented implementation choice."""
    if method == "clip":
        return np.clip(trial, lower, upper)
    # Midpoint repair moves out-of-range coordinates halfway toward the
    # violated bound from their feasible parent's coordinate.
    trial = np.where(trial < lower, (lower + parent) / 2.0, trial)
    trial = np.where(trial > upper, (upper + parent) / 2.0, trial)
    return trial


def omcde(
    objective: BatchObjective,
    dim: int,
    bounds: Union[tuple[float, float], tuple[Array, Array]] = (-100.0, 100.0),
    *,
    max_evals: Optional[int] = None,
    population_size: Optional[int] = None,
    min_population_size: int = 4,
    archive_ratio: float = 1.4,
    memory_size: int = 6,
    stage_switch: float = 0.75,
    boundary_repair: str = "midpoint",
    seed: Optional[int] = None,
) -> OMCDEResult:
    """Minimize a batch objective using OMCDE.

    Args:
        objective: Vectorized objective: (N, dim) -> (N,) fitnesses.
        dim: Decision-variable count.
        bounds: Scalar (lower, upper), or length-dim bound vectors.
        max_evals: Total row evaluations; defaults to 10_000 * dim.
        population_size: Initial N, defaults to 18 * dim.
        min_population_size: Paper specifies 4; non-triplet sizes are supported.
        archive_ratio: Max archive size / current population size.
        memory_size: History length H.
        stage_switch: Fraction of total evaluations spent in Phase 1.
        boundary_repair: 'midpoint' (default) or 'clip'.
        seed: Random seed for repeatable independent runs.

    Important: Unlike Algorithm 1's FES=N_init line, both original and
    opposite initial candidate evaluations are charged to the budget.
    """
    if not isinstance(dim, (int, np.integer)) or dim < 1:
        raise ValueError("dim must be a positive integer")
    max_evals = 10_000 * dim if max_evals is None else int(max_evals)
    n_init = 18 * dim if population_size is None else int(population_size)
    n_min = int(min_population_size)
    if n_min < 4 or n_init < n_min:
        raise ValueError("require initial population >= minimum >= 4")
    if max_evals < 2 * n_init:
        raise ValueError("max_evals must be >= 2 * initial population (OBL)")
    if memory_size < 1 or archive_ratio < 0:
        raise ValueError("memory_size must be >= 1; archive_ratio >= 0")
    if not (0.0 < stage_switch < 1.0):
        raise ValueError("stage_switch must be between 0 and 1")
    if boundary_repair not in ("midpoint", "clip"):
        raise ValueError("boundary_repair must be 'midpoint' or 'clip'")
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (dim,)).copy()
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (dim,)).copy()
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly below upper bound")

    start = perf_counter()
    rng = np.random.default_rng(seed)
    population = rng.uniform(lower, upper, size=(n_init, dim))
    opposition = lower + upper - population   # Eq. 10
    original_fitness = _evaluate(objective, population)
    opposition_fitness = _evaluate(objective, opposition)
    # Eq. 11 chooses the better complete population, not candidate-wise.
    if opposition_fitness.sum() < original_fitness.sum():
        population, fitness = opposition, opposition_fitness
    else:
        fitness = original_fitness
    nfev = 2 * n_init  # Account for both complete initial fitness evaluations.
    archive = np.empty((0, dim), dtype=float)
    memory_f = np.full(memory_size, 0.5)
    memory_cr = np.full(memory_size, 0.5)
    memory_slot = 0
    best_idx = int(np.argmin(fitness))
    global_best = population[best_idx].copy()
    global_fitness = float(fitness[best_idx])

    history = {
        "nfev": [nfev],
        "best_f": [global_fitness],
        "population_size": [len(population)],
        "diversity": [float(np.mean(np.std(population, axis=0)))],
        "phase": ["initialization"],
    }
    generation = 0
    while nfev < max_evals:
        # Eq. 18: shrink before generating the next generation.
        target_size = int(
            np.rint(n_init + (n_min - n_init) * (nfev / max_evals))
        )
        target_size = max(n_min, min(len(population), target_size))
        if target_size < len(population):
            keep = np.argsort(fitness)[:target_size]
            population, fitness = population[keep], fitness[keep]
        n = len(population)
        archive_limit = int(np.floor(archive_ratio * n))
        if len(archive) > archive_limit:
            selected = rng.choice(len(archive), size=archive_limit, replace=False)
            archive = archive[selected]

        factors, crossover_rates = _controls(rng, memory_f, memory_cr, n)
        greedy = nfev > stage_switch * max_evals
        mutants = _competitive_mutants(
            rng, population, fitness, factors, archive, global_best, greedy
        )

        # In a final partial generation, evaluate only the remaining budget.
        eval_count = min(n, max_evals - nfev)
        targets = (
            np.arange(n)
            if eval_count == n
            else rng.choice(n, size=eval_count, replace=False)
        )
        mask = rng.random((eval_count, dim)) < crossover_rates[targets, None]
        mandatory_dim = rng.integers(0, dim, size=eval_count)
        mask[np.arange(eval_count), mandatory_dim] = True
        parents = population[targets]
        trials = np.where(mask, mutants[targets], parents)
        trials = _repair_bounds(trials, parents, lower, upper, boundary_repair)
        trial_fitness = _evaluate(objective, trials)
        nfev += eval_count
        improved = trial_fitness < fitness[targets]
        success_ids = targets[improved]

        if len(success_ids):
            improvement = fitness[success_ids] - trial_fitness[improved]
            # Archive successful offspring's *replaced parents* (JADE convention).
            archive = np.concatenate((archive, population[success_ids].copy()))
            if len(archive) > archive_limit:
                selected = rng.choice(
                    len(archive), size=archive_limit, replace=False
                )
                archive = archive[selected]
            population[success_ids] = trials[improved]
            fitness[success_ids] = trial_fitness[improved]

            mean_f = _weighted_lehmer(factors[success_ids], improvement)
            mean_cr = _weighted_lehmer(crossover_rates[success_ids], improvement)
            if mean_f is not None:
                memory_f[memory_slot] = 0.5 * memory_f[memory_slot] + 0.5 * mean_f
            if mean_cr is not None:
                memory_cr[memory_slot] = 0.5 * memory_cr[memory_slot] + 0.5 * mean_cr
            memory_slot = (memory_slot + 1) % memory_size

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < global_fitness:
            global_fitness = float(fitness[best_idx])
            global_best = population[best_idx].copy()
        generation += 1
        history["nfev"].append(nfev)
        history["best_f"].append(global_fitness)
        history["population_size"].append(n)
        history["diversity"].append(float(np.mean(np.std(population, axis=0))))
        history["phase"].append("greedy" if greedy else "competitive")

    return OMCDEResult(
        x=global_best,
        fun=global_fitness,
        nfev=nfev,
        nit=generation,
        elapsed_seconds=perf_counter() - start,
        history=history,
        final_population=population.copy(),
        final_fitness=fitness.copy(),
    )
