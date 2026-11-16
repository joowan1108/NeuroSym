#!/usr/bin/env bash
# [image build] RLDX-1 v1.0.2 env from the official uv.lock (torch 2.7.0, CUDA 12.6 runtime).
# flash-attn is compiled from source here, which needs the CUDA devel toolkit (nvcc).
# This env targets RTX 40-series; RTX 50-series would need the CUDA 12.8 build of torch.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$SRC"
export MAX_JOBS="${MAX_JOBS:-3}"   # limits memory use of the flash-attn build

git clone --branch "$RLDX_TAG" https://github.com/RLWRLD/RLDX-1.git "$SRC/RLDX-1"
cd "$SRC/RLDX-1"
uv sync --python 3.10
uv pip install --python .venv/bin/python -e .
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -c "
import torch, flash_attn
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper
print('[ok] rldx env: torch', torch.__version__)"
