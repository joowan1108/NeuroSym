#!/usr/bin/env bash
# [instance] RoboCasa assets (~10 GB) and the two checkpoints (~24 GB). Safe to re-run.
# HF_TOKEN, if needed, comes from the Vast account environment variables.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$CKPT" "$HF_HOME"

if [[ ! -f $WORK/.robocasa_assets_done ]]; then
    (yes y || true) | "$VENVS/sim/bin/python" -m robocasa.scripts.download_kitchen_assets
    touch "$WORK/.robocasa_assets_done"
fi
"$VENVS/xr1/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365', local_dir='$XR1_MODEL')"
"$SRC/RLDX-1/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('RLWRLD/RLDX-1-FT-RC365', local_dir='$RLDX_MODEL')"
echo "[ok] downloads done"
