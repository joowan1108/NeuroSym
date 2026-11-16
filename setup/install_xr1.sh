#!/usr/bin/env bash
# [image build] Xiaomi-Robotics-1 deploy env (docs/DEPLOYMENT.md of the official repo).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$SRC" "$VENVS"

git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1 "$SRC/Xiaomi-Robotics-1"
git -C "$SRC/Xiaomi-Robotics-1" checkout "$XR1_COMMIT"

uv venv "$VENVS/xr1" --python 3.12
source "$VENVS/xr1/bin/activate"
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==4.57.1 accelerate "imageio[ffmpeg]" tqdm scipy pillow numpy
# Prebuilt wheel listed in the official deployment guide (py3.12 / torch 2.8 / CUDA 12).
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

cd "$SRC/Xiaomi-Robotics-1/eval_robocasa365"
PYTHONPATH="$SRC/Xiaomi-Robotics-1/deploy" python -c "
import torch, flash_attn, transformers, entry
print('[ok] xr1 env: torch', torch.__version__, 'transformers', transformers.__version__)"
