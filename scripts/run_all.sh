#!/usr/bin/env bash
# Full experiment: for each model and task
#   1) evaluate the base policy            -> results/<model>/<task>/eval_base_iid.json
#   2) train DSRL (time and steps logged)  -> results/<model>/<task>/train_summary.json
#   3) evaluate the best steered policy    -> results/<model>/<task>/eval_dsrl.json
#
# Everything is overridable from the environment, e.g.
#   MODELS=xr1 TASKS=CoffeeSetupMug N_ENVS=8 STEPS_CoffeeSetupMug=50000 bash scripts/run_all.sh
# Finished stages are skipped, so the script can simply be re-run after an interruption.
# It can be started right after logging in: it waits for the RoboCasa assets, then for each
# model's checkpoint, while setup/download.sh keeps downloading in the background.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../setup/env.sh"
cd "$PROJECT"

MODELS="${MODELS:-xr1 rldx}"
TASKS="${TASKS:-CoffeeSetupMug GetToastedBread SteamInMicrowave}"
SPLIT="${SPLIT:-pretrain}"
N_ENVS="${N_ENVS:-4}"
SEED="${SEED:-7}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
PORT="${PORT:-10086}"
RESULTS="${RESULTS:-$PROJECT/results}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
# RL budget in environment steps (task horizons: 600 / 3000 / 2100).
STEPS_CoffeeSetupMug="${STEPS_CoffeeSetupMug:-100000}"
STEPS_GetToastedBread="${STEPS_GetToastedBread:-300000}"
STEPS_SteamInMicrowave="${STEPS_SteamInMicrowave:-200000}"

PY="$VENVS/sim/bin/python"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
SERVER_PID=""
MON_PID=""
MODEL_T0=0
stop_server() {
    # `wait` on a killed child returns 143; "|| true" keeps set -e from aborting the run here.
    if [[ -n $SERVER_PID ]]; then   # wall time this model held the GPU (summed over re-runs)
        printf '%s\t%s\n' "$(date -Is)" "$((SECONDS - MODEL_T0))" >> "$RESULTS/$model/model_wall_time.tsv"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n $MON_PID ]]; then
        kill "$MON_PID" 2>/dev/null || true
        wait "$MON_PID" 2>/dev/null || true
    fi
    SERVER_PID=""; MON_PID=""
    return 0
}
trap stop_server EXIT

# stage <out_dir> <name> <command...>: runs the command, logs it, and appends
# "<name>\t<seconds>" to <out_dir>/stage_times.tsv. Must not be called as the
# condition of if/||/&& (bash would then ignore errors inside it).
stage() {
    local out=$1 name=$2 t0=$SECONDS
    shift 2
    "$@" 2>&1 | tee -a "$out/log.txt"
    printf '%s\t%s\n' "$name" "$((SECONDS - t0))" >> "$out/stage_times.tsv"
}

mkdir -p "$RESULTS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$RESULTS/gpu.txt"
echo "[run] GPU: $(cat "$RESULTS/gpu.txt")"

wait_for_download assets
for model in $MODELS; do
    wait_for_download "$model"
    mkdir -p "$RESULTS/$model"
    MODEL_T0=$SECONDS
    bash scripts/run_server.sh "$model" "$PORT" >"$RESULTS/$model/server.log" 2>&1 &
    SERVER_PID=$!
    # GPU utilisation (%) and memory (MiB) every 10 s while this model runs.
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 \
        >> "$RESULTS/$model/gpu_usage.csv" &
    MON_PID=$!
    echo "[run] started $model server (pid $SERVER_PID); log: $RESULTS/$model/server.log"

    for task in $TASKS; do
        out="$RESULTS/$model/$task"
        mkdir -p "$out"
        common=(--model "$model" --task "$task" --port "$PORT" --split "$SPLIT"
                --n-envs "$N_ENVS" --seed "$SEED" --out-dir "$out")
        steps_var="STEPS_$task"

        if [[ ! -f $out/eval_base_iid.json ]]; then
            stage "$out" base_eval "$PY" -u -m dsrl.eval "${common[@]}" --episodes "$EVAL_EPISODES"
        fi

        if [[ ! -f $out/train.done ]]; then
            # shellcheck disable=SC2086
            stage "$out" train "$PY" -u -m dsrl.train_dsrl "${common[@]}" \
                --total-env-steps "${!steps_var}" $EXTRA_TRAIN_ARGS
            touch "$out/train.done"
        fi

        if [[ ! -f $out/eval_dsrl.json ]]; then
            stage "$out" dsrl_eval "$PY" -u -m dsrl.eval "${common[@]}" --episodes "$EVAL_EPISODES" \
                --agent "$out/best.pt"
        fi
    done
    stop_server
done

"$PY" scripts/summarize.py "$RESULTS"
