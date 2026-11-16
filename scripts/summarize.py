"""Collect results/<model>/<task>/*.json into tables (markdown on stdout + summary.csv + gpu_summary.csv).

Per task:  success rates, RL wall time / env steps, and the wall time of every stage.
Per model: which GPU, how many hours the model held it, GPU utilisation and peak memory.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(result) -> str:
    if result is None:
        return "-"
    return f"{100 * result['success_rate']:.1f} ({result['successes']}/{result['num_episodes']})"


def num(value, fmt="{:.0f}") -> str:
    return "-" if value is None else fmt.format(value)


def read_tsv_seconds(path: Path) -> dict[str, float]:
    """name<TAB>seconds lines -> {name: total seconds} (re-runs are summed)."""
    out: dict[str, float] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                out[parts[0]] = out.get(parts[0], 0.0) + float(parts[1])
    return out


def gpu_stats(path: Path):
    util, mem = [], []
    if path.exists():
        for row in csv.reader(path.open()):
            try:
                util.append(float(row[1]))
                mem.append(float(row[2]))
            except (IndexError, ValueError):
                continue
    if not util:
        return None, None
    return sum(util) / len(util), max(mem)


def table(header, rows) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    gpu_file = root / "gpu.txt"
    gpu = gpu_file.read_text().strip().splitlines()[0].split(",")[0].strip() if gpu_file.exists() else "?"

    header = [
        "model", "task", "base SR %", "DSRL SR %", "RL time (s)", "RL env steps (#)",
        "RL VLA queries (#)", "RL episodes (#)", "time to best (s)", "steps to best (#)",
        "steps/episode base", "steps/episode DSRL", "wall time all stages (h)",
    ]
    rows = []
    for task_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
        base = load(task_dir / "eval_base_iid.json")
        dsrl = load(task_dir / "eval_dsrl.json")
        train = load(task_dir / "train_summary.json") or {}
        if not (base or dsrl or train):
            continue
        best = train.get("to_best") or {}
        stages = read_tsv_seconds(task_dir / "stage_times.tsv")
        rows.append([
            task_dir.parent.name, task_dir.name, pct(base), pct(dsrl),
            num(train.get("rl_wall_time_s")), num(train.get("rl_env_steps")),
            num(train.get("rl_decisions")), num(train.get("rl_episodes")),
            num(best.get("rl_wall_time_s")), num(best.get("rl_env_steps")),
            num((base or {}).get("mean_steps_to_success")), num((dsrl or {}).get("mean_steps_to_success")),
            num(sum(stages.values()) / 3600 if stages else None, "{:.2f}"),
        ])
    table(header, rows)

    gpu_header = [
        "model", "GPU", "GPU hours (h)", "model loading (h)", "GPU hours excl. loading (h)",
        "mean GPU util (%)", "peak GPU memory (MiB)",
    ]
    gpu_rows = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        wall = read_tsv_seconds(model_dir / "model_wall_time.tsv")
        if not wall:
            continue
        hours = sum(wall.values()) / 3600
        load_h = sum(read_tsv_seconds(model_dir / "model_load_time.tsv").values()) / 3600
        util, peak = gpu_stats(model_dir / "gpu_usage.csv")
        gpu_rows.append([
            model_dir.name, f"1x {gpu}", f"{hours:.2f}", f"{load_h:.2f}", f"{hours - load_h:.2f}",
            num(util, "{:.0f}"), num(peak),
        ])
    if gpu_rows:
        print()
        table(gpu_header, gpu_rows)
        total = sum(float(r[2]) for r in gpu_rows)
        work = sum(float(r[4]) for r in gpu_rows)
        print(f"\ntotal: 1x {gpu}, {total:.2f} h ({work:.2f} h excluding model loading)")

    with (root / "summary.csv").open("w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    with (root / "gpu_summary.csv").open("w", newline="") as f:
        csv.writer(f).writerows([gpu_header] + gpu_rows)
    print(f"\nwrote {root / 'summary.csv'} and {root / 'gpu_summary.csv'}")


if __name__ == "__main__":
    main()
