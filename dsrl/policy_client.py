"""Sim-side handle on a steerable VLA server (pi_dp^W in the DSRL paper)."""

from __future__ import annotations

import numpy as np

from dsrl.wire import RpcClient


class SteerablePolicy:
    def __init__(self, host: str, port: int):
        self.rpc = RpcClient(host, port)
        self.info = self.rpc.call({"cmd": "info"})
        self.name: str = self.info["model"]
        self.horizon, self.dim = (int(x) for x in self.info["noise_shape"])
        self.active = np.asarray(self.info["active_dims"], dtype=np.int64)
        self.frame_deltas = [int(d) for d in self.info["frame_deltas"]]
        self.default_exec_steps = int(self.info["default_exec_steps"])
        self.infer_time = 0.0
        self.n_queries = 0

    @property
    def steer_dim(self) -> int:
        """Dimension of the latent-noise action the RL agent controls (per action step)."""
        return len(self.active)

    def iid_noise(self, batch: int, rng: np.random.Generator) -> np.ndarray:
        """w ~ N(0, I) over the whole chunk: the unmodified base policy."""
        return rng.standard_normal((batch, self.horizon, self.dim)).astype(np.float32)

    def expand(self, w: np.ndarray) -> np.ndarray:
        """Single-step noise (B, d) repeated along the chunk axis (DSRL appendix C.5)."""
        noise = np.zeros((w.shape[0], self.horizon, self.dim), dtype=np.float32)
        noise[:, :, self.active] = w[:, None, :]
        return noise

    def act(self, obs: list[dict], noise: np.ndarray, reset: list[bool], session_ids: list[str]) -> np.ndarray:
        request = {
            "cmd": "act",
            "obs": [{"frames": o["frames"], "instruction": o["instruction"]} for o in obs],
            "noise": noise,
            "reset": reset,
            "session_ids": session_ids,
        }
        response = self.rpc.call(request)
        self.infer_time += float(response["infer_time"])
        self.n_queries += len(obs)
        return response["actions"]

    def close(self) -> None:
        self.rpc.close()
