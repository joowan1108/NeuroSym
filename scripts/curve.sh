#!/usr/bin/env bash
# Learning curve: re-evaluate every checkpoint saved during DSRL training (ckpt_<env steps>.pt)
# on the same held-out seeds as the final evaluation.
#   MODELS=xr1 TASKS=CoffeeSetupMug CURVE_EPISODES=50 bash scripts/curve.sh
# Writes results/<model>/<task>/eval_curve_<steps>.json and curve.csv. Finished points are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../setup/env.sh"
cd "$PROJECT"
MODELS="${MODELS:-xr1 rldx}"
TASKS="${TASKS:-CoffeeSetupMug GetToastedBread SteamInMicrowave}"
SPLIT="${SPLIT:-pretrain}"; N_ENVS="${N_ENVS:-4}"; SEED="${SEED:-7}"; PORT="${PORT:-10086}"
CURVE_EPISODES="${CURVE_EPISODES:-50}"
RESULTS="${RESULTS:-$PROJECT/results}"
PY="$VENVS/sim/bin/python"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
SERVER_PID=""
trap '[[ -n $SERVER_PID ]] && kill "$SERVER_PID" 2>/dev/null || true' EXIT

for model in $MODELS; do
    bash scripts/run_server.sh "$model" "$PORT" >"$RESULTS/$model/server_curve.log" 2>&1 &
    SERVER_PID=$!
    until grep -q "ready:" "$RESULTS/$model/server_curve.log" 2>/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "[x] $model server exited while loading" >&2; exit 1; }
        sleep 2
    done
    for task in $TASKS; do
        out="$RESULTS/$model/$task"
        for ckpt in $(ls "$out"/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n); do
            steps=$(basename "$ckpt" .pt); steps=${steps#ckpt_}
            [[ -f $out/eval_curve_$steps.json ]] && continue
            "$PY" -u -m dsrl.eval --model "$model" --task "$task" --port "$PORT" --split "$SPLIT" \
                --n-envs "$N_ENVS" --seed "$SEED" --out-dir "$out" --episodes "$CURVE_EPISODES" \
                --agent "$ckpt" --tag "curve_$steps"
        done
        "$PY" - "$out" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
base = out / "eval_base_iid.json"
if base.exists():
    b = json.loads(base.read_text()); rows.append((0, b["successes"], b["num_episodes"]))
for f in out.glob("eval_curve_*.json"):
    d = json.loads(f.read_text()); rows.append((int(f.stem.split("_")[-1]), d["successes"], d["num_episodes"]))
rows.sort()
(out / "curve.csv").write_text("env_steps,successes,episodes,success_rate\n" + "".join(f"{s},{k},{n},{k/n:.3f}\n" for s, k, n in rows))
print(f"[curve] {out}: " + "  ".join(f"{s}: {k}/{n}" for s, k, n in rows))
PY
    done
    kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""
done
