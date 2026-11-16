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
stop_server() { [[ -n $SERVER_PID ]] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }
trap stop_server EXIT

wait_for_download assets
for model in $MODELS; do
    wait_for_download "$model"
    mkdir -p "$RESULTS/$model"
    bash scripts/run_server.sh "$model" "$PORT" >"$RESULTS/$model/server.log" 2>&1 &
    SERVER_PID=$!
    echo "[run] started $model server (pid $SERVER_PID); log: $RESULTS/$model/server.log"

    for task in $TASKS; do
        out="$RESULTS/$model/$task"
        mkdir -p "$out"
        common=(--model "$model" --task "$task" --port "$PORT" --split "$SPLIT"
                --n-envs "$N_ENVS" --seed "$SEED" --out-dir "$out")
        steps_var="STEPS_$task"

        [[ -f $out/eval_base_iid.json ]] || \
            "$PY" -u -m dsrl.eval "${common[@]}" --episodes "$EVAL_EPISODES" 2>&1 | tee -a "$out/log.txt"

        # shellcheck disable=SC2086
        [[ -f $out/train.done ]] || {
            "$PY" -u -m dsrl.train_dsrl "${common[@]}" --total-env-steps "${!steps_var}" \
                $EXTRA_TRAIN_ARGS 2>&1 | tee -a "$out/log.txt"
            touch "$out/train.done"
        }

        [[ -f $out/eval_dsrl.json ]] || \
            "$PY" -u -m dsrl.eval "${common[@]}" --episodes "$EVAL_EPISODES" --agent "$out/best.pt" \
                2>&1 | tee -a "$out/log.txt"
    done
    stop_server
done

"$PY" scripts/summarize.py "$RESULTS"
