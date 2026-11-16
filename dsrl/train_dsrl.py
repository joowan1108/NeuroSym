"""DSRL-SAC training of a latent-noise policy that steers a frozen VLA on one RoboCasa365 task.

Reports, in <out-dir>/train_summary.json:
  rl_wall_time_s   wall-clock seconds spent in RL (rollouts + gradient updates; periodic
                   evaluation time is tracked separately and excluded)
  rl_env_steps     environment steps consumed by RL (includes the warm-up rollouts)
  rl_decisions     VLA queries (= latent-MDP transitions), rl_episodes, grad_steps
  evals            success rate of the deterministic steered policy at each checkpoint,
                   with the RL steps/seconds spent up to that point
  to_best / to_target   RL steps and seconds needed to first reach the best / target success
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from dsrl.common import EVAL_SEED_OFFSET, add_common_args, connect, evaluate, seed_stream, write_json
from dsrl.envs import PROP_DIM
from dsrl.rollout import run_episodes
from dsrl.sac import DSRLAgent, ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--total-env-steps", type=int, required=True)
    parser.add_argument("--warmup-episodes", type=int, default=8, help="base-policy rollouts with w~N(0,I)")
    parser.add_argument("--utd", type=int, default=20, help="gradient steps per decision")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--noise-mode", default="chunk", choices=("chunk", "repeat", "repeat_norm"),
                        help="chunk: steer the whole chunk noise; repeat: one step noise repeated (pi_0 trick)")
    parser.add_argument("--magnitude", type=float, default=3.0, help="b_W: latent actions live in [-b_W, b_W]^d")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-critics", type=int, default=10)
    parser.add_argument("--init-alpha", type=float, default=0.01,
                        help="initial SAC temperature (kept small: an entropy bonus rewards not finishing)")
    parser.add_argument("--buffer-size", type=int, default=40_000)
    parser.add_argument("--eval-every", type=int, default=None, help="env steps; default total/4")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--target-success", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cudnn.benchmark = True

    policy, pool, exec_steps = connect(args)
    d = policy.steer_dim
    agent = DSRLAgent(
        act_dim=d, magnitude=args.magnitude, hidden=args.hidden,
        n_critics=args.n_critics, gamma=args.gamma, init_alpha=args.init_alpha, device=args.device,
    )
    agent.noise_mode = args.noise_mode  # stored in the checkpoint so evaluation expands w the same way
    buffer = ReplayBuffer(args.buffer_size, (9, 64, 64), PROP_DIM, d)
    eval_every = args.eval_every or max(1, args.total_env_steps // 4)
    eval_seeds = [args.seed + EVAL_SEED_OFFSET + i for i in range(args.eval_episodes)]
    print(
        f"[train] {args.model}/{args.task}: horizon={pool.horizon} exec_steps={exec_steps} "
        f"latent dim={d} ({args.noise_mode}; noise {policy.horizon}x{policy.dim})", flush=True,
    )

    stats = dict(env_steps=0, decisions=0, episodes=0, successes=0, rl_time=0.0, update_time=0.0,
                 sim_time=0.0, eval_time=0.0, pending_updates=0)
    recent = deque(maxlen=20)
    log_file = (out / "train_log.jsonl").open("a")
    evals: list[dict] = []
    train_seed = iter(range(args.seed, args.seed + 10**9))
    state = dict(warm=True, limit=0)

    def choose_train(obs):
        if state["warm"]:
            w = rng.standard_normal((len(obs), d)).astype(np.float32)
            w = np.clip(w, -args.magnitude, args.magnitude)
        else:
            w = agent.act(np.stack([o["img"] for o in obs]), np.stack([o["prop"] for o in obs]))
        return w, policy.expand(w)

    def choose_eval(obs):
        w = agent.act(np.stack([o["img"] for o in obs]), np.stack([o["prop"] for o in obs]), deterministic=True)
        return w, policy.expand(w)

    def on_step(obs, w, result):
        # Sparse reward of the paper's pi_0 experiments: -1 until success, then 0 (terminal).
        success = result["success"]
        buffer.add(obs["img"], obs["prop"], w, 0.0 if success else -1.0,
                   result["img"], result["prop"], float(success))
        stats["env_steps"] += result["n_steps"]
        stats["decisions"] += 1
        stats["sim_time"] += result["sim_time"]
        stats["pending_updates"] += args.utd

    def on_episode(ep):
        stats["episodes"] += 1
        stats["successes"] += int(ep["success"])
        recent.append(ep["success"])
        if state["warm"] and stats["episodes"] >= args.warmup_episodes:
            agent.set_prop_stats(buffer.prop[: buffer.size])
            state["warm"] = False
            stats["pending_updates"] = 0
            print(f"[train] warm-up done: {stats['successes']}/{stats['episodes']} base successes", flush=True)
        log_file.write(json.dumps(dict(kind="episode", env_steps=stats["env_steps"], **ep)) + "\n")
        log_file.flush()

    def after_events():
        if state["warm"]:
            return
        tic = time.time()
        info = {}
        for _ in range(stats["pending_updates"]):
            info = agent.update(buffer.sample(args.batch_size, agent.device))
        stats["pending_updates"] = 0
        stats["update_time"] += time.time() - tic
        if info and stats["decisions"] % 50 < pool.n:
            print(
                f"[train] steps={stats['env_steps']} ep={stats['episodes']} "
                f"recent_sr={np.mean(recent) if recent else 0:.2f} q={info['q']:.2f} "
                f"alpha={info['alpha']:.3f} rl_time={stats['rl_time_live']():.0f}s", flush=True,
            )

    def summary() -> dict:
        best = max(evals, key=lambda e: (e["success_rate"], -e["rl_env_steps"])) if evals else None
        hit = None
        if args.target_success is not None:
            hit = next((e for e in evals if e["success_rate"] >= args.target_success), None)
        pick = lambda e: None if e is None else {  # noqa: E731
            k: e[k] for k in ("success_rate", "rl_env_steps", "rl_wall_time_s", "rl_decisions", "rl_episodes")
        }
        return dict(
            model=args.model, task=args.task, split=args.split, seed=args.seed,
            horizon=pool.horizon, exec_steps=exec_steps, latent_dim=d,
            rl_wall_time_s=stats["rl_time"], rl_env_steps=stats["env_steps"],
            rl_decisions=stats["decisions"], rl_episodes=stats["episodes"],
            grad_steps=agent.grad_steps,
            train_success_rate_overall=stats["successes"] / max(1, stats["episodes"]),
            time_breakdown_s=dict(
                vla_inference=policy.infer_time, sim_cpu_sum=stats["sim_time"],
                sac_updates=stats["update_time"], periodic_eval=stats["eval_time"],
            ),
            evals=[{k: v for k, v in e.items() if k != "episodes"} for e in evals],
            to_best=pick(best), target_success=args.target_success, to_target=pick(hit),
            config=vars(args),
        )

    segment_start = [time.time()]
    stats["rl_time_live"] = lambda: stats["rl_time"] + time.time() - segment_start[0]
    next_eval = eval_every
    try:
        while stats["env_steps"] < args.total_env_steps:
            state["limit"] = min(args.total_env_steps, next_eval)
            segment_start[0] = time.time()
            run_episodes(
                pool, policy, exec_steps,
                next_seed=seed_stream(train_seed), choose_noise=choose_train,
                on_step=on_step, on_episode=on_episode, after_events=after_events,
                stop=lambda: stats["env_steps"] >= state["limit"],
            )
            stats["rl_time"] += time.time() - segment_start[0]

            tic = time.time()
            infer_before = policy.infer_time
            if state["warm"]:  # budget ended inside warm-up
                agent.set_prop_stats(buffer.prop[: buffer.size])
            result = evaluate(pool, policy, exec_steps, eval_seeds, choose_eval)
            policy.infer_time = infer_before  # keep the RL inference tally free of eval time
            stats["eval_time"] += time.time() - tic
            result.update(rl_env_steps=stats["env_steps"], rl_wall_time_s=stats["rl_time"],
                          rl_decisions=stats["decisions"], rl_episodes=stats["episodes"])
            best_so_far = max((e["success_rate"] for e in evals), default=-1.0)
            evals.append(result)
            agent.save(out / "last.pt")
            if result["success_rate"] > best_so_far:
                agent.save(out / "best.pt")
            print(
                f"[eval] steps={stats['env_steps']} rl_time={stats['rl_time']:.0f}s "
                f"success={result['successes']}/{result['num_episodes']}", flush=True,
            )
            write_json(out / "train_summary.json", summary())
            next_eval += eval_every
    finally:
        log_file.close()
        pool.close()
        policy.close()
    print(json.dumps({k: v for k, v in summary().items() if k not in ("evals", "config")}, indent=2))


if __name__ == "__main__":
    main()
