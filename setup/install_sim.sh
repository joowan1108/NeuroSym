#!/usr/bin/env bash
# [image build] Simulator env: RoboCasa365 + robosuite + the DSRL learner (PyTorch).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$SRC" "$VENVS"

uv venv "$VENVS/sim" --python 3.11
source "$VENVS/sim/bin/activate"
uv pip install setuptools wheel

git clone https://github.com/ARISE-Initiative/robosuite "$SRC/robosuite"
git clone https://github.com/robocasa/robocasa "$SRC/robocasa"
git -C "$SRC/robocasa" checkout "$ROBOCASA_COMMIT"
uv pip install -e "$SRC/robosuite" --config-settings editable_mode=compat
uv pip install -e "$SRC/robocasa" --config-settings editable_mode=compat

# robocasa's dependencies already pull in torch 2.7.1 (CUDA 12.6 build), so this is a no-op
# that only pins the version. Fine for RTX 40-series; RTX 50-series would need
# --reinstall-package torch --reinstall-package torchvision to get the CUDA 12.8 build.
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

(yes y || true) | python -m robocasa.scripts.setup_macros
MUJOCO_GL=egl python -c "
import torch, cv2, gymnasium, robosuite, robocasa
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
for t in ('CoffeeSetupMug', 'GetToastedBread', 'SteamInMicrowave'):
    assert t in TASK_SET_REGISTRY['target50'], t
    print(t, 'horizon', get_task_horizon(t))
print('[ok] sim env: torch', torch.__version__)"
