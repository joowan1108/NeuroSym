# Shared paths and pins. Source this from every script:  source setup/env.sh
#   /opt        baked into the Docker image: Python environments and third-party repos
#   /workspace  created on the instance: this repo, checkpoints, HF cache, results
export WORK="${WORK:-/workspace}"
export PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SRC="${SRC:-/opt/src}"
export VENVS="${VENVS:-/opt/venvs}"
export CKPT="${CKPT:-$WORK/checkpoints}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export RESULTS="${RESULTS:-$WORK/results}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/opt/uv-python}"
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV   # the Vast base image auto-activates its own venv

# Pinned upstream revisions (the ones this code was written against).
export ROBOCASA_COMMIT="425684d332f135f5b30a40b321f52d15a0fe9250"   # robocasa master, 2026-11-12
export XR1_COMMIT="0dd7aef8dc87296246aae812a1f59ccb708e5546"        # XiaomiRobotics/Xiaomi-Robotics-1
export RLDX_TAG="v1.0.2"                                            # RLWRLD/RLDX-1

export XR1_MODEL="$CKPT/Xiaomi-Robotics-1-RoboCasa365"
export RLDX_MODEL="$CKPT/RLDX-1-FT-RC365"

# Headless MuJoCo rendering on the GPU.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
