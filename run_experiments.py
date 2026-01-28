import os
import json
from itertools import product

import matplotlib.pyplot as plt

from gcp_ga import load_col_graph, run_ga, GAConfig, pretty_solution_summary


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def save_fitness_plot(history, out_path: str, title: str) -> None:
    plt.figure()
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness so far")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # Put your datasets here (download from CMU COLOR instances, as recommended in the assignment :contentReference[oaicite:1]{index=1})
    datasets = [
        # ("small",  "data/DSJC1000.1.col.b"),
        #  ("medium", "data/queen8_12.col.b"),
        ("large",  "data/le450_5d.col.b"),
    ]

    out_dir = "results"
    ensure_dir(out_dir)

    # 6+ combinations (you can expand easily)
    selections = ["tournament", "roulette"]
    crossovers = ["one_point", "uniform"]
    mutations = ["random_reset", "greedy"]

    # We'll pick exactly 6 combos by choosing a subset:
    combos = [
        ("tournament", "one_point", "random_reset"),
        ("tournament", "uniform", "random_reset"),
        ("tournament", "one_point", "greedy"),
        ("roulette", "one_point", "random_reset"),
        ("roulette", "uniform", "random_reset"),
        ("roulette", "one_point", "greedy"),
    ]

    # Common hyperparams (tune per size if desired)
    base = dict(
        generations=2000,
        elite_count=2,
        crossover_rate=0.9,
        mutation_rate=0.05,
        tournament_k=3,
        conflict_weight=50.0,
        patience=150,
        seed=42,
    )

    for tag, path in datasets:
        V, edges = load_col_graph(path)

        # scale pop size with V (simple rule)
        pop_size = min(500, max(80, 10 * V))

        dataset_dir = os.path.join(out_dir, tag)
        ensure_dir(dataset_dir)

        summary_rows = []
        best_run = None

        for i, (sel, cross, mut) in enumerate(combos, start=1):
            cfg = GAConfig(
                population_size=pop_size,
                selection=sel,
                crossover=cross,
                mutation=mut,
                max_colors=V,  # safe upper bound
                **base,
            )
            res = run_ga(V, edges, cfg)

            print(f"[{tag}] run {i}/6:", pretty_solution_summary(res))

            row = {
                "dataset": tag,
                "path": path,
                "V": V,
                "E": len(edges),
                "run": i,
                "selection": sel,
                "crossover": cross,
                "mutation": mut,
                "pop_size": pop_size,
                "generations_ran": res.generations_ran,
                "time_seconds": res.time_seconds,
                "best_fitness": res.best_fitness,
                "best_conflicts": res.best_conflicts,
                "best_colors_used": res.best_colors_used,
            }
            summary_rows.append(row)

            if best_run is None or res.best_fitness < best_run.best_fitness:
                best_run = res

            # Save per-run JSON
            run_json = os.path.join(dataset_dir, f"run_{i}.json")
            with open(run_json, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

        # Save overall summary
        summary_path = os.path.join(dataset_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)

        # Plot best run
        if best_run is not None:
            plot_path = os.path.join(dataset_dir, "best_fitness_evolution.png")
            save_fitness_plot(
                best_run.fitness_history,
                plot_path,
                title=f"{tag}: best fitness evolution (V={V}, E={len(edges)})",
            )
            print(f"[{tag}] best plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
