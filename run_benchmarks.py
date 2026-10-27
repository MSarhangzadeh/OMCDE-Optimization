"""Reproduce OMCDE on five standard analytic functions in 30 dimensions.

These benchmarks are unshifted and unrotated; they are not CEC 2017.
Run: python run_benchmarks.py
"""
from pathlib import Path
import csv
import json
import numpy as np
from omcde import omcde

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
D = 30
MAX_EVALS = 10_000 * D
SEEDS = (0, 1, 2, 3, 4)

def sphere(x): return np.sum(x * x, axis=1)
def rastrigin(x): return 10 * x.shape[1] + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=1)
def rosenbrock(x): return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)
def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x * x, axis=1))) - np.exp(np.mean(np.cos(2 * np.pi * x), axis=1)) + 20 + np.e
def griewank(x):
    idx = np.sqrt(np.arange(1, x.shape[1] + 1))
    return 1 + np.sum(x * x, axis=1) / 4000 - np.prod(np.cos(x / idx), axis=1)

BENCHMARKS = {
    "Sphere": (sphere, (-100, 100)),
    "Rastrigin": (rastrigin, (-5.12, 5.12)),
    "Rosenbrock": (rosenbrock, (-5, 10)),
    "Ackley": (ackley, (-32.768, 32.768)),
    "Griewank": (griewank, (-600, 600)),
}

def main():
    rows, histories = [], {}
    for name, (fn, bounds) in BENCHMARKS.items():
        histories[name] = []
        for seed in SEEDS:
            result = omcde(fn, dim=D, bounds=bounds, max_evals=MAX_EVALS, seed=seed)
            rows.append(dict(
                function=name, dim=D, seed=seed, max_evals=MAX_EVALS,
                final_fitness=float(result.fun),
                elapsed_seconds=float(result.elapsed_seconds),
                generations=result.nit,
                final_population_size=len(result.final_population),
            ))
            xp = np.linspace(0, MAX_EVALS, 201)
            yp = np.interp(xp, np.asarray(result.history["nfev"]), np.asarray(result.history["best_f"]))
            histories[name].append([float(z) for z in yp])
            print(f"{name:11} seed={seed} error={result.fun:.8g} time={result.elapsed_seconds:.2f}s", flush=True)
    with (RESULTS / "benchmark_d30_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name in BENCHMARKS:
        fitness = np.asarray([r["final_fitness"] for r in rows if r["function"] == name])
        times = np.asarray([r["elapsed_seconds"] for r in rows if r["function"] == name])
        summary.append(dict(
            function=name, dim=D, runs=len(fitness), evaluations_per_run=MAX_EVALS,
            best=float(fitness.min()), mean=float(fitness.mean()),
            std=float(fitness.std(ddof=1)), median=float(np.median(fitness)),
            worst=float(fitness.max()), mean_runtime_seconds=float(times.mean()),
        ))
    with (RESULTS / "benchmark_d30_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (RESULTS / "benchmark_d30_curves.json").open("w") as f:
        json.dump(dict(
            dim=D, max_evals=MAX_EVALS, seeds=list(SEEDS),
            fractions=[float(i / 200) for i in range(201)],
            history=histories, summary=summary,
        ), f, separators=(",", ":"))
    return summary

if __name__ == "__main__":
    main()
