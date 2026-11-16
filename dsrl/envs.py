"""RoboCasa365 environments in worker processes, stepped one action chunk at a time.

Each worker owns one `gym.make("robocasa/<Task>", split=..., seed=...)` env (the
same construction both official evaluators use) and answers two commands:

  ("reset", seed)          -> observation payload
  ("step", actions[k,12])  -> payload after executing up to k actions; stops
                              early on success or when the task horizon is hit.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from collections import deque
from multiprocessing.connection import wait

import numpy as np

CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
)
LANGUAGE_KEY = "annotation.human.task_description"
PROP_DIM = 16  # 3 + 4 + 2 + 3 + 4


def _worker(conn, cfg: dict) -> None:
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import cv2
        import gymnasium as gym
        import robocasa  # noqa: F401  (registers robocasa/<Task>)
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from robocasa.utils.env_utils import convert_action

        task = cfg["task"]
        env = gym.make(f"robocasa/{task}", split=cfg["split"], seed=cfg["seed"])
        horizon = cfg["horizon"] or int(get_task_horizon(task))
        deltas = cfg["frame_deltas"]
        size = cfg["rl_image_size"]
        history: deque = deque(maxlen=max(0, -min(deltas)) + 1)
        instruction = ""
        steps = 0

        def slim(obs):
            out = {k: np.ascontiguousarray(obs[k], dtype=np.uint8) for k in CAMERA_KEYS}
            out.update({k: np.asarray(obs[k], dtype=np.float32) for k in STATE_KEYS})
            return out

        def payload():
            items = list(history)
            cur = items[-1]
            # Missing history at episode start repeats the first frame (as both evaluators do).
            frames = [items[max(0, len(items) - 1 + d)] for d in deltas]
            img = np.concatenate(
                [cv2.resize(cur[k], (size, size), interpolation=cv2.INTER_AREA) for k in CAMERA_KEYS],
                axis=-1,
            ).transpose(2, 0, 1)
            prop = np.concatenate([cur[k].reshape(-1) for k in STATE_KEYS]).astype(np.float32)
            return {
                "frames": frames,
                "instruction": instruction,
                "img": np.ascontiguousarray(img),
                "prop": prop,
            }

        conn.send(("ready", {"horizon": horizon}))
        while True:
            cmd, arg = conn.recv()
            if cmd == "close":
                break
            if cmd == "reset":
                obs, _ = env.reset(seed=int(arg))
                instruction = str(obs[LANGUAGE_KEY])
                history.clear()
                history.append(slim(obs))
                steps = 0
                conn.send(("ok", payload()))
            elif cmd == "step":
                tic = time.time()
                success = False
                executed = 0
                for action in np.asarray(arg, dtype=np.float32):
                    obs, _, _, _, info = env.step(convert_action(action))
                    history.append(slim(obs))
                    steps += 1
                    executed += 1
                    success = bool(info.get("success", False))
                    if success or steps >= horizon:
                        break
                out = payload()
                out.update(
                    n_steps=executed,
                    success=success,
                    timeout=(not success) and steps >= horizon,
                    sim_time=time.time() - tic,
                )
                conn.send(("ok", out))
        env.close()
    except Exception:
        conn.send(("error", traceback.format_exc()))


class EnvPool:
    """Asynchronous pool: send commands, then collect whatever has finished."""

    def __init__(
        self,
        n_envs: int,
        task: str,
        split: str,
        seed: int,
        frame_deltas: list[int],
        horizon: int | None = None,
        rl_image_size: int = 64,
    ):
        ctx = mp.get_context("spawn")  # EGL contexts do not survive fork
        self.n = n_envs
        self.conns, self.procs = [], []
        for i in range(n_envs):
            parent, child = ctx.Pipe()
            cfg = dict(
                task=task,
                split=split,
                seed=seed + i,
                horizon=horizon,
                frame_deltas=frame_deltas,
                rl_image_size=rl_image_size,
            )
            proc = ctx.Process(target=_worker, args=(child, cfg), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        self._index = {id(c): i for i, c in enumerate(self.conns)}
        self.busy: dict[int, str] = {}
        self.horizon = 0
        for i, conn in enumerate(self.conns):
            kind, data = conn.recv()
            if kind == "error":
                raise RuntimeError(f"env worker {i} failed to start:\n{data}")
            self.horizon = int(data["horizon"])

    def send_reset(self, i: int, seed: int) -> None:
        self.conns[i].send(("reset", seed))
        self.busy[i] = "reset"

    def send_step(self, i: int, actions: np.ndarray) -> None:
        self.conns[i].send(("step", actions))
        self.busy[i] = "step"

    def recv(self) -> list[tuple[int, str, dict]]:
        """Block for at least one result, and for every in-flight *step* (keeps
        policy batches together); resets are slow and are left running."""
        events = []
        while self.busy:
            stepping = any(kind == "step" for kind in self.busy.values())
            timeout = None if (not events or stepping) else 0
            ready = wait([self.conns[i] for i in self.busy], timeout)
            if not ready:
                break
            for conn in ready:
                i = self._index[id(conn)]
                status, data = conn.recv()
                if status == "error":
                    raise RuntimeError(f"env worker {i} crashed:\n{data}")
                events.append((i, self.busy.pop(i), data))
        return events

    def drain(self) -> None:
        while self.busy:
            self.recv()

    def close(self) -> None:
        self.drain()
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
