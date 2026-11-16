#!/usr/bin/env bash
# [instance] GPU, EGL rendering and simulator speed check. Waits for the RoboCasa assets.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
ldconfig -p | grep -q libEGL_nvidia || echo "[!] libEGL_nvidia missing: the template needs -e NVIDIA_DRIVER_CAPABILITIES=all"
"$VENVS/xr1/bin/python" -c "import torch; print('xr1  cuda:', torch.cuda.is_available(), torch.__version__)"
"$SRC/RLDX-1/.venv/bin/python" -c "import torch; print('rldx cuda:', torch.cuda.is_available(), torch.__version__)"
wait_for_download assets
"$VENVS/sim/bin/python" - <<'PY'
import time
import gymnasium as gym, numpy as np, robocasa, torch
from robocasa.utils.env_utils import convert_action
print("sim  cuda:", torch.cuda.is_available(), torch.__version__)
env = gym.make("robocasa/CoffeeSetupMug", split="pretrain", seed=0)
tic = time.time(); obs, _ = env.reset(seed=1); reset_s = time.time() - tic
tic = time.time()
for _ in range(20):
    obs, _, _, _, info = env.step(convert_action(np.zeros(12, np.float32)))
print(f"reset {reset_s:.1f}s, {20 / (time.time() - tic):.1f} env steps/s (3 cameras)")
print({k: getattr(v, "shape", v) for k, v in obs.items()})
env.close()
print("[ok] simulator works")
PY
