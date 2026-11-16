"""Collect results/<model>/<task>/*.json into one table (markdown on stdout + summary.csv)."""

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


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    header = [
        "model", "task", "base SR %", "DSRL SR %", "RL time (s)", "RL env steps (#)",
        "RL VLA queries (#)", "RL episodes (#)", "time to best (s)", "steps to best (#)",
        "steps/episode base", "steps/episode DSRL",
    ]
    rows = []
    for task_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
        base = load(task_dir / "eval_base_iid.json")
        dsrl = load(task_dir / "eval_dsrl.json")
        train = load(task_dir / "train_summary.json")
        if not (base or dsrl or train):
            continue
        best = (train or {}).get("to_best") or {}
        rows.append([
            task_dir.parent.name, task_dir.name, pct(base), pct(dsrl),
            num((train or {}).get("rl_wall_time_s")), num((train or {}).get("rl_env_steps")),
            num((train or {}).get("rl_decisions")), num((train or {}).get("rl_episodes")),
            num(best.get("rl_wall_time_s")), num(best.get("rl_env_steps")),
            num((base or {}).get("mean_steps_to_success")), num((dsrl or {}).get("mean_steps_to_success")),
        ])

    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    with (root / "summary.csv").open("w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(f"\nwrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
