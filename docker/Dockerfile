# Vast.ai image for the DSRL x RoboCasa365 experiments (RTX 4090).
#
#   docker build --platform linux/amd64 -f docker/Dockerfile -t joowan1108/dsrl-robocasa365:v1 .
#
# In the image:   OS packages, CUDA 12.4 devel toolkit, the three Python environments
#                 (/opt/venvs/sim, /opt/venvs/xr1, /opt/src/RLDX-1/.venv) and the pinned
#                 third-party repos they are installed from (/opt/src).
# Not in the image: this project's code (GitHub, cloned by the on-start script),
#                 RoboCasa assets and checkpoints (downloaded on the instance), tokens.
FROM vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py314-2026-10-21

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs tmux htop curl wget ca-certificates build-essential cmake ninja-build ffmpeg \
        libegl1 libgl1 libgles2 libglvnd0 libglx0 libopengl0 libosmesa6 libglfw3 \
        libusb-1.0-0 libhidapi-dev \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA EGL vendor file (the driver library itself is mounted by the NVIDIA runtime
# when NVIDIA_DRIVER_CAPABILITIES includes "graphics").
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy

# One layer per environment so a change in one does not rebuild the others.
COPY setup/env.sh /opt/build/setup/env.sh
COPY setup/install_xr1.sh /opt/build/setup/
RUN bash /opt/build/setup/install_xr1.sh && uv cache clean
COPY setup/install_sim.sh /opt/build/setup/
RUN bash /opt/build/setup/install_sim.sh && uv cache clean
COPY setup/install_rldx.sh /opt/build/setup/
RUN bash /opt/build/setup/install_rldx.sh && uv cache clean

WORKDIR /workspace
