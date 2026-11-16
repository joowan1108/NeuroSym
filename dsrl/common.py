"""Helpers shared by train_dsrl.py and eval.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dsrl.envs import EnvPool
from dsrl.policy_client import SteerablePolicy
from dsrl.rollout import run_episodes

TASKS = ("CoffeeSetupMug", "GetToastedBread", "SteamInMicrowave")
EVAL_SEED_OFFSET = 1_000_000  # evaluation scenes never overlap with training seeds


def add_common_args(parser) -> None:
    parser.add_argument("--model", required=True, choices=("xr1", "rldx"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--split", default="pretrain", choices=("pretrain", "target"))
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=None, help="default: robocasa get_task_horizon")
    parser.add_argument("--exec-steps", type=int, default=None, help="actions executed per query")
    parser.add_argument("--out-dir", required=True)


def connect(args) -> tuple[SteerablePolicy, EnvPool, int]:
    policy = SteerablePolicy(args.host, args.port, getattr(args, "noise_mode", "chunk"), args.seed)
    if policy.name != args.model:
        raise RuntimeError(f"server on port {args.port} serves {policy.name!r}, not {args.model!r}")
    exec_steps = args.exec_steps or policy.default_exec_steps
    if exec_steps > policy.horizon:
        raise ValueError(f"--exec-steps {exec_steps} exceeds the action chunk ({policy.horizon})")
    pool = EnvPool(
        n_envs=args.n_envs,
        task=args.task,
        split=args.split,
        seed=args.seed,
        frame_deltas=policy.frame_deltas,
        horizon=args.horizon,
    )
    return policy, pool, exec_steps


def seed_stream(seeds):
    it = iter(seeds)
    return lambda: next(it, None)


def evaluate(pool, policy, exec_steps, seeds, choose_noise) -> dict:
    """Roll out one episode per seed and summarise."""
    episodes: list[dict] = []
    run_episodes(
        pool, policy, exec_steps,
        next_seed=seed_stream(seeds),
        choose_noise=choose_noise,
        on_episode=episodes.append,
    )
    episodes.sort(key=lambda e: e["seed"])
    n = len(episodes)
    successes = sum(e["success"] for e in episodes)
    rate = successes / n
    win_steps = [e["steps"] for e in episodes if e["success"]]
    return dict(
        num_episodes=n,
        successes=successes,
        success_rate=rate,
        success_rate_stderr=math.sqrt(rate * (1 - rate) / n),
        mean_episode_steps=float(np.mean([e["steps"] for e in episodes])),
        mean_steps_to_success=float(np.mean(win_steps)) if win_steps else None,
        episodes=episodes,
    )


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)
