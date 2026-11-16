"""Sim-side handle on a steerable VLA server (pi_dp^W in the DSRL paper)."""

from __future__ import annotations

import numpy as np

from dsrl.wire import RpcClient


class SteerablePolicy:
    """noise_mode:
      chunk   the agent chooses the whole initial noise of the chunk, (H x d_real) numbers.
              With w ~ N(0, I) this is exactly the base policy.
      repeat  one d_real-dimensional step noise repeated along the chunk (DSRL appendix C.5,
              used for pi_0). Much smaller action space, but on XR-1 the repeated noise is
              off-distribution: base success dropped from 50% to 0/20 on CoffeeSetupMug.
    Only the real action dims are steered; padded dims always get fresh N(0, 1) noise
    (they are masked out in XR-1 and carry plain noise at training time in RLDX-1).
    """

    def __init__(self, host: str, port: int, noise_mode: str = "chunk", seed: int = 0):
        self.rpc = RpcClient(host, port)
        self.info = self.rpc.call({"cmd": "info"})
        self.name: str = self.info["model"]
        self.horizon, self.dim = (int(x) for x in self.info["noise_shape"])
        self.steer = np.arange(int(self.info["action_dim"]), dtype=np.int64)
        self.noise_mode = noise_mode
        self.frame_deltas = [int(d) for d in self.info["frame_deltas"]]
        self.default_exec_steps = int(self.info["default_exec_steps"])
        self.rng = np.random.default_rng(seed)
        self.infer_time = 0.0
        self.n_queries = 0

    @property
    def step_dim(self) -> int:
        return len(self.steer)

    @property
    def steer_dim(self) -> int:
        """Dimension of the latent-noise action the RL agent controls."""
        return self.step_dim * (self.horizon if self.noise_mode == "chunk" else 1)

    def iid_noise(self, batch: int, rng: np.random.Generator) -> np.ndarray:
        """w ~ N(0, I) over the whole chunk: the unmodified base policy."""
        return rng.standard_normal((batch, self.horizon, self.dim)).astype(np.float32)

    def expand(self, w: np.ndarray) -> np.ndarray:
        """Latent action -> initial noise (B, H, D), according to `noise_mode`.

        chunk        w is (B, H * step_dim), rescaled to norm sqrt(H * step_dim)
        repeat       w is (B, step_dim), used as is and repeated along the chunk
        repeat_norm  like repeat, but rescaled to norm sqrt(step_dim) first

        The rescaling keeps the noise at the radius where N(0, I) concentrates. The flow model
        only ever saw noise of that norm: with small-norm noise XR-1 drops from 50% to 15-20%
        success, and SAC shrinks its noise towards the mean as it becomes confident."""
        noise = self.rng.standard_normal((w.shape[0], self.horizon, self.dim)).astype(np.float32)
        if self.noise_mode != "repeat":
            w = w * (np.sqrt(w.shape[1]) / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-6))
        if w.shape[1] == self.step_dim:
            noise[:, :, self.steer] = w[:, None, :]
        else:
            noise[:, :, self.steer] = w.reshape(w.shape[0], self.horizon, self.step_dim)
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
