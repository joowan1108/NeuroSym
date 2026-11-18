"""Steerable inference server for Xiaomi-Robotics-1 (RoboCasa365 checkpoint).

Run inside the XR-1 deploy env (torch 2.8 / transformers 4.57.1 / flash-attn).
Pre/post-processing is NOT re-implemented: we reuse `EvalClient.infer` from the
official `eval_robocasa365/entry.py` and only swap its socket client for an
in-process one that can inject the initial flow-matching noise.
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

ROBOT_TYPE = "robocasa365"


class LocalModelClient:
    """Drop-in for deploy/client.py:Client that calls the model in-process."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.noise: torch.Tensor | None = None

    def __call__(self, **data):
        robot_type = data.get("task_id")
        model = self.model
        # Same tensor placement as deploy/server.py.
        data = {
            key: (
                value.to(device=model.device, dtype=model.dtype)
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value.to(device=model.device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in data.items()
        }
        noise = self.noise
        m = 1 if noise is None else noise.shape[0]
        if m > 1:
            # Several noises for the same observation (DSRL-NA): run the VLM once and repeat its outputs
            # (KV cache, position ids, attention mask) along the batch, so the unchanged model.forward runs the
            # action expert on all m noises at once.
            data["state"] = data["state"].repeat_interleave(m, dim=0)
            data["action_mask"] = data["action_mask"].repeat_interleave(m, dim=0)
            vlm = model.vlm
            vlm_forward = vlm.forward

            def repeated_vlm_forward(*args, **kwargs):
                out = vlm_forward(*args, **kwargs)
                out.past_key_values.batch_repeat_interleave(m)
                out.position_ids = out.position_ids.repeat_interleave(m, dim=1)
                out.attention_mask = out.attention_mask.repeat_interleave(m, dim=0)
                return out

            vlm.forward = repeated_vlm_forward
        if noise is not None and tuple(noise.shape) != tuple(data["action_mask"].shape):
            raise ValueError(
                f"noise shape {tuple(noise.shape)} != action_mask shape "
                f"{tuple(data['action_mask'].shape)}"
            )
        try:
            with inject_initial_noise(noise, "randn_like") as state:
                outputs = model(**data)
        finally:
            if m > 1:
                del model.vlm.forward   # back to the class method
        require_single_hit(state, noise, "XR-1")
        decoded = self.processor.decode_action(outputs.actions.cpu(), robot_type=robot_type)
        self.last_all = decoded   # all m decoded chunks; EvalClient.infer only keeps the first
        return decoded

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--xr1-repo", required=True, help="clone of XiaomiRobotics/Xiaomi-Robotics-1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--obs-history", type=int, default=4)
    parser.add_argument("--obs-interval", type=int, default=2)
    parser.add_argument("--crop-ratio", type=float, default=0.95)
    parser.add_argument("--replan-steps", type=int, default=16)
    args = parser.parse_args()

    repo = Path(args.xr1_repo).resolve()
    sys.path.insert(0, str(repo / "deploy"))
    sys.path.insert(0, str(repo / "eval_robocasa365"))
    import entry  # official evaluator (helpers + EvalClient)
    from transformers import AutoModel, AutoProcessor

    print("[xr1] loading model ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = (
        AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    model.eval()

    local = LocalModelClient(model, processor)
    client = entry.EvalClient.__new__(entry.EvalClient)  # skip the socket-connecting __init__
    client.client = local
    client.processor = processor
    client.robot_type = ROBOT_TYPE
    client.crop_ratio = args.crop_ratio

    mask = processor.get_action_mask(ROBOT_TYPE)[0].float()  # (H, D)
    horizon, dim = int(mask.shape[0]), int(mask.shape[1])
    # Masked dims are multiplied by 0 at every denoising step, so only these matter.
    active_dims = [int(i) for i in torch.nonzero(mask.amax(dim=0) > 0).flatten()]
    deltas = [-(args.obs_history - 1 - i) * args.obs_interval for i in range(args.obs_history)]
    info = {
        "model": "xr1",
        "noise_shape": [horizon, dim],
        "active_dims": active_dims,
        "frame_deltas": deltas,
        "action_dim": entry.ACTION_DIM,
        "default_exec_steps": args.replan_steps,
    }
    print(f"[xr1] ready: {info}", flush=True)

    def handle(request: dict) -> dict:
        if request["cmd"] == "info":
            return info
        if request["cmd"] != "act":
            raise ValueError(f"unknown cmd {request['cmd']!r}")
        noise = request.get("noise")
        actions = []
        tic = time.time()
        for i, obs in enumerate(request["obs"]):
            frames = obs["frames"]
            states = np.stack([entry.observation_to_state(frame) for frame in frames])
            images = {
                key: np.stack([np.ascontiguousarray(frame[key], dtype=np.uint8) for frame in frames])
                for key in entry.CAMERA_KEYS
            }
            n_i = None if noise is None else torch.from_numpy(np.asarray(noise[i], np.float32))
            multi = n_i is not None and n_i.dim() == 3          # (m, H, D): several noises, one observation
            local.noise = None if n_i is None else (n_i if multi else n_i[None])
            try:
                first = client.infer(states, images, obs["instruction"])
                actions.append(local.last_all[:, :, : first.shape[-1]].float().numpy() if multi else first)
            finally:
                local.noise = None
        torch.cuda.synchronize()
        return {"actions": np.stack(actions).astype(np.float32), "infer_time": time.time() - tic}

    serve(handle, args.host, args.port)


if __name__ == "__main__":
    main()
