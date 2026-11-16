"""Steerable inference server for RLDX-1-FT-RC365.

Run inside the RLDX-1 (v1.0.2) uv environment. Uses the official
`RLDXSimPolicyWrapper(RLDXPolicy(...))` pipeline unchanged and only injects the
initial flow-matching noise of the action head.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dsrl.noise_hook import inject_initial_noise, require_single_hit  # noqa: E402
from dsrl.wire import serve  # noqa: E402

# Order of the 12-D RoboCasa365 action vector (robocasa.utils.env_utils.convert_action).
ACTION_KEYS = [
    "end_effector_position",
    "end_effector_rotation",
    "gripper_close",
    "base_motion",
    "control_mode",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=8, help="official eval default")
    args = parser.parse_args()

    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

    print("[rldx] loading model ...", flush=True)
    policy = RLDXPolicy(
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        model_path=args.model_path,
        device=args.device,
    )
    sim_policy = RLDXSimPolicyWrapper(policy)

    configs = policy.modality_configs
    video_keys = list(configs["video"].modality_keys)
    state_keys = list(configs["state"].modality_keys)
    video_deltas = [int(d) for d in configs["video"].delta_indices]
    state_deltas = [int(d) for d in configs["state"].delta_indices]
    language_key = list(configs["language"].modality_keys)[0]
    action_keys = list(configs["action"].modality_keys)
    if action_keys != ACTION_KEYS:
        raise RuntimeError(f"unexpected action keys {action_keys}; expected {ACTION_KEYS}")
    frame_deltas = sorted(set(video_deltas) | set(state_deltas))
    video_idx = [frame_deltas.index(d) for d in video_deltas]
    state_idx = [frame_deltas.index(d) for d in state_deltas]

    action_model = policy.model.action_model
    horizon = int(action_model.config.action_horizon)
    dim = int(action_model.action_dim)

    info = {
        "model": "rldx",
        "noise_shape": [horizon, dim],
        # Padded dims also carried noise during training, so every dim is steerable.
        "active_dims": list(range(dim)),
        "frame_deltas": frame_deltas,
        "action_dim": 12,
        "default_exec_steps": args.n_action_steps,
    }
    print(f"[rldx] ready: {info} (memory={policy.use_memory})", flush=True)

    def handle(request: dict) -> dict:
        if request["cmd"] == "info":
            return info
        if request["cmd"] != "act":
            raise ValueError(f"unknown cmd {request['cmd']!r}")
        obs_list = request["obs"]
        batch = len(obs_list)
        observation = {}
        for key in video_keys:
            observation[f"video.{key}"] = np.stack(
                [
                    np.stack([np.asarray(o["frames"][j][f"video.{key}"], np.uint8) for j in video_idx])
                    for o in obs_list
                ]
            )
        for key in state_keys:
            observation[f"state.{key}"] = np.stack(
                [
                    np.stack(
                        [np.asarray(o["frames"][j][f"state.{key}"], np.float32).reshape(-1) for j in state_idx]
                    )
                    for o in obs_list
                ]
            )
        observation[language_key] = [str(o["instruction"]) for o in obs_list]
        options = {
            "reset_memory": [bool(x) for x in request["reset"]],
            "session_ids": [str(x) for x in request["session_ids"]],
        }

        noise = request.get("noise")
        noise_t = None
        if noise is not None:
            noise_t = torch.from_numpy(np.asarray(noise, np.float32))
            if tuple(noise_t.shape) != (batch, horizon, dim):
                raise ValueError(f"noise shape {tuple(noise_t.shape)} != {(batch, horizon, dim)}")

        tic = time.time()
        with inject_initial_noise(noise_t, "randn") as state:
            action, _ = sim_policy.get_action(observation, options)
        require_single_hit(state, noise_t, "RLDX-1")
        torch.cuda.synchronize()
        actions = np.concatenate([action[f"action.{k}"] for k in ACTION_KEYS], axis=-1)
        if actions.shape[-1] != 12:
            raise RuntimeError(f"expected 12-D actions, got {actions.shape}")
        return {"actions": actions.astype(np.float32), "infer_time": time.time() - tic}

    serve(handle, args.host, args.port)


if __name__ == "__main__":
    main()
