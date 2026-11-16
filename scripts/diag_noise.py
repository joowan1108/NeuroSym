"""Diagnostic: what does the VLA output for differently shaped initial noise on the SAME observation?

  python scripts/diag_noise.py --model xr1 --task CoffeeSetupMug   (policy server must be running)

Prints, per noise type, statistics of the returned action chunk (H, 12): magnitude, how much the
actions vary along the chunk, and the distance to the mean i.i.d.-noise action.
"""

import argparse

import numpy as np

from dsrl.envs import EnvPool
from dsrl.policy_client import SteerablePolicy

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--task", default="CoffeeSetupMug")
parser.add_argument("--port", type=int, default=10086)
parser.add_argument("--seeds", type=int, nargs="+", default=[1000007, 1000011])
parser.add_argument("--draws", type=int, default=8)
args = parser.parse_args()

policy = SteerablePolicy("127.0.0.1", args.port)
pool = EnvPool(1, args.task, "pretrain", 7, policy.frame_deltas)
rng = np.random.default_rng(0)
H, D, d = policy.horizon, policy.dim, policy.step_dim


def make(kind):
    noise = rng.standard_normal((1, H, D)).astype(np.float32)
    if kind == "repeat":
        noise[:, :, :d] = rng.standard_normal((1, 1, d))
    elif kind == "repeat_norm":
        w = rng.standard_normal((1, 1, d)); noise[:, :, :d] = w * np.sqrt(d) / np.linalg.norm(w)
    elif kind == "zero":
        noise[:, :, :d] = 0
    elif kind == "small":
        noise[:, :, :d] *= 0.2
    return noise


for seed in args.seeds:
    pool.send_reset(0, seed)
    obs = pool.recv()[0][2]
    out = {}
    for kind in ("iid", "repeat", "repeat_norm", "small", "zero"):
        acts = [policy.act([obs], make(kind), [True], ["diag"])[0] for _ in range(args.draws)]
        out[kind] = np.stack(acts)  # (draws, H, 12)
    ref = out["iid"].mean(0)
    print(f"\nseed {seed}: {obs['instruction']}")
    print(f"{'noise':12s} {'|a| arm(0:6)':>13s} {'std along chunk':>16s} {'std across draws':>17s} {'dist to iid mean':>17s} {'gripper mean':>13s}")
    for kind, a in out.items():
        arm = a[:, :, :6]
        print(f"{kind:12s} {np.abs(arm).mean():13.3f} {arm.std(axis=1).mean():16.3f} {arm.std(axis=0).mean():17.3f} "
              f"{np.abs(arm - ref[None, :, :6]).mean():17.3f} {a[:, :, 6].mean():13.3f}")
pool.close()
