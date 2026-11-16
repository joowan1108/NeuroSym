#!/usr/bin/env bash
# Start the steerable policy server for one model:  scripts/run_server.sh {xr1|rldx} [port]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../setup/env.sh"
MODEL="${1:?usage: run_server.sh xr1|rldx [port]}"
PORT="${2:-10086}"
cd "$PROJECT"

case "$MODEL" in
    xr1)
        exec "$VENVS/xr1/bin/python" -u dsrl/servers/xr1_server.py \
            --model-path "$XR1_MODEL" --xr1-repo "$SRC/Xiaomi-Robotics-1" --port "$PORT"
        ;;
    rldx)
        export NO_ALBUMENTATIONS_UPDATE=1
        exec "$SRC/RLDX-1/.venv/bin/python" -u dsrl/servers/rldx_server.py \
            --model-path "$RLDX_MODEL" --port "$PORT"
        ;;
    *) echo "unknown model $MODEL" >&2; exit 1 ;;
esac
