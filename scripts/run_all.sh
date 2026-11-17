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
# PARALLEL=1 runs all tasks of a model at the same time against one policy server. Faster overall,
# but the tasks compete for the server, so "RL time" is longer than when a task runs alone.
PARALLEL="${PARALLEL:-0}"
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

# run_task <task>: base evaluation -> DSRL training -> evaluation of the steered policy
run_task() {
    local task=$1 out="$RESULTS/$model/$1" steps_var="STEPS_$1"
    local common=(--model "$model" --task "$task" --port "$PORT" --split "$SPLIT"
                  --n-envs "$N_ENVS" --seed "$SEED" --out-dir "$out")
    [[ -n ${FIXED_SCENE:-} ]] && common+=(--fixed-scene "$FIXED_SCENE")   # one kitchen, e.g. FIXED_SCENE=12,34
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
    # Wait for the "ready:" line the servers print after loading the model, so loading time is
    # recorded separately and a crash during loading stops the run right away.
    until grep -q "ready:" "$RESULTS/$model/server.log" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[x] $model server exited while loading; see $RESULTS/$model/server.log" >&2
            tail -20 "$RESULTS/$model/server.log" >&2
            exit 1
        fi
        if (( SECONDS - MODEL_T0 > 3600 )); then
            echo "[x] $model server not ready after 1 h; see $RESULTS/$model/server.log" >&2
            exit 1
        fi
        sleep 2
    done
    printf '%s\t%s\n' "$(date -Is)" "$((SECONDS - MODEL_T0))" >> "$RESULTS/$model/model_load_time.tsv"
    echo "[run] $model server ready after $((SECONDS - MODEL_T0)) s"

    pids=()
    for task in $TASKS; do
        mkdir -p "$RESULTS/$model/$task"
        if (( PARALLEL )); then   # all tasks of this model at once, sharing the policy server
            run_task "$task" > "$RESULTS/$model/$task/run.log" 2>&1 &
            pids+=("$!")
            echo "[run] $model/$task started in background (log: $RESULTS/$model/$task/run.log)"
        else
            run_task "$task"
        fi
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if (( failed )); then
        echo "[x] a task of $model failed; see $RESULTS/$model/*/run.log" >&2
        exit 1
    fi
    stop_server
done

"$PY" scripts/summarize.py "$RESULTS"
