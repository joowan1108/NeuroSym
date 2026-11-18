"""Event loop that runs episodes of the latent-action MDP M^W on an EnvPool.

One "decision" = one query of the VLA with a chosen initial noise, followed by
`exec_steps` environment steps (the action chunk is treated as a single action,
as in DSRL appendix C.1).
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

import os

from dsrl.envs import EnvPool
from dsrl.policy_client import SteerablePolicy


# unique per client process: several experiments may share one policy server
SESSION_PREFIX = f"p{os.getpid()}"


def run_episodes(
    pool: EnvPool,
    policy: SteerablePolicy,
    exec_steps: int,
    next_seed: Callable[[], int | None],
    choose_noise: Callable[[list[dict]], tuple[np.ndarray | None, np.ndarray]],
    on_step: Callable[[dict, np.ndarray | None, dict], None] | None = None,
    on_episode: Callable[[dict], None] | None = None,
    after_events: Callable[[], None] | None = None,
    stop: Callable[[], bool] = lambda: False,
) -> None:
    """Runs until `next_seed` is exhausted or `stop()` is true.

    choose_noise(obs_list) -> (w or None, noise[B, H, D]).
    on_step(obs, w, result, env_index) is called once per finished decision.
    Episodes still in progress when `stop()` fires are abandoned.
    """
    pool.drain()
    ready: dict[int, dict] = {}
    inflight: dict[int, dict] = {}
    episodes: dict[int, dict] = {}

    def start_episode(i: int) -> None:
        seed = next_seed()
        if seed is None:
            return
        episodes[i] = dict(seed=int(seed), steps=0, decisions=0, first=True, t0=time.time())
        pool.send_reset(i, int(seed))

    for i in range(pool.n):
        start_episode(i)

    while True:
        if ready and not stop():
            ids = sorted(ready)
            obs = [ready.pop(i) for i in ids]
            w, noise = choose_noise(obs)
            actions = policy.act(
                obs,
                noise,
                reset=[episodes[i]["first"] for i in ids],
                session_ids=[f"{SESSION_PREFIX}-env{i}" for i in ids],
            )
            for j, i in enumerate(ids):
                episodes[i]["first"] = False
                # actions (B, H, A), or (B, M, H, A) when several noises were queried per observation
                # (DSRL-NA): the first one is executed, all of them are handed to on_step.
                multi = actions.ndim == 4
                inflight[i] = dict(obs=obs[j], w=None if w is None else w[j], acts=actions[j] if multi else None)
                pool.send_step(i, actions[j, 0, :exec_steps] if multi else actions[j, :exec_steps])
        if not pool.busy:
            break
        for i, kind, data in pool.recv():
            if kind == "reset":
                ready[i] = data
                continue
            sent = inflight.pop(i)
            ep = episodes[i]
            ep["steps"] += data["n_steps"]
            ep["decisions"] += 1
            if on_step is not None:
                if sent["acts"] is not None:
                    data = dict(data, vla_actions=sent["acts"])
                on_step(sent["obs"], sent["w"], data, i)
            if data["success"] or data["timeout"]:
                if on_episode is not None:
                    on_episode(
                        dict(
                            seed=ep["seed"],
                            success=bool(data["success"]),
                            steps=ep["steps"],
                            decisions=ep["decisions"],
                            wall=time.time() - ep["t0"],
                        )
                    )
                if not stop():
                    start_episode(i)
            else:
                ready[i] = data
        if after_events is not None:
            after_events()
