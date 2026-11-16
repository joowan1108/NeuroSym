"""Evaluate a VLA on one RoboCasa365 task, either as-is or steered by a DSRL agent.

  --agent none  --noise iid     the unmodified base policy (fresh w ~ N(0, I) per chunk)
  --agent none  --noise repeat  base policy with one N(0, I) step-noise repeated along the
                                chunk (sanity check of the latent parameterisation DSRL uses)
  --agent <ckpt>                deterministic DSRL-steered policy

Base and steered runs use the same evaluation seeds, so they see the same scenes.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from dsrl.common import EVAL_SEED_OFFSET, add_common_args, connect, evaluate, write_json


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--agent", default="none")
    parser.add_argument("--noise", default="iid", choices=("iid", "repeat"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--tag", default=None, help="output file stem; default base_<noise> / dsrl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy, pool, exec_steps = connect(args)
    rng = np.random.default_rng(args.seed)

    if args.agent != "none":
        from dsrl.sac import DSRLAgent

        agent = DSRLAgent.load(args.agent, device=args.device)
        if agent.cfg["act_dim"] not in (policy.step_dim, policy.step_dim * policy.horizon):
            raise ValueError("agent latent dim does not match this model")
        # older checkpoints have no noise_mode: infer it from the latent dimension
        policy.noise_mode = agent.noise_mode or ("chunk" if agent.cfg["act_dim"] > policy.step_dim else "repeat")

        def choose(obs):
            w = agent.act(np.stack([o["img"] for o in obs]), np.stack([o["prop"] for o in obs]), deterministic=True)
            return w, policy.expand(w)

        tag = args.tag or "dsrl"
    elif args.noise == "iid":
        choose = lambda obs: (None, policy.iid_noise(len(obs), rng))  # noqa: E731
        tag = args.tag or "base_iid"
    else:

        def choose(obs):
            w = rng.standard_normal((len(obs), policy.step_dim)).astype(np.float32)
            policy.noise_mode = "repeat"
            return w, policy.expand(w)

        tag = args.tag or "base_repeat"

    seeds = [args.seed + EVAL_SEED_OFFSET + i for i in range(args.episodes)]
    tic = time.time()
    try:
        result = evaluate(pool, policy, exec_steps, seeds, choose)
    finally:
        pool.close()
    result.update(
        model=args.model, task=args.task, split=args.split, tag=tag, agent=args.agent,
        horizon=pool.horizon, exec_steps=exec_steps,
        eval_wall_time_s=time.time() - tic,
        mean_vla_latency_s=policy.infer_time / max(1, policy.n_queries),
    )
    policy.close()
    write_json(f"{args.out_dir}/eval_{tag}.json", result)
    print(
        f"[eval:{tag}] {args.model}/{args.task}: {result['successes']}/{result['num_episodes']} "
        f"= {100 * result['success_rate']:.1f}% (+-{100 * result['success_rate_stderr']:.1f}), "
        f"steps-to-success={result['mean_steps_to_success']}"
    )


if __name__ == "__main__":
    main()
