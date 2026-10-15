#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-/nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260922--w1_5_0p5--signw0p05--residual0p1/checkpoint-100000/}"
DEVICE="${DEVICE:-cuda:0}"
WEB_PORT="${WEB_PORT:-9090}"
PROMPT_VERSION="${PROMPT_VERSION:-1}"
TARGET_AREA="${TARGET_AREA:-1}"

# First oracle experiment: use simulator GT bbox and found/not-found status.
# Change this to pred after the GT residual experiment has been evaluated.
SIGN_CONDITIONING_MODE="${SIGN_CONDITIONING_MODE:-gt_bbox_status}"

# E3-A must keep the grounded token enabled. Other modes remain available for
# controlled ablations through an explicit environment override.
SIGN_ABLATION_MODE="${SIGN_ABLATION_MODE:-normal}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

resolve_latest_checkpoint() {
    local root="$1"
    if [[ -f "$root/config.json" && -f "$root/processor_config.json" ]]; then
        printf '%s\n' "$root"
        return
    fi
    [[ -d "$root" ]] || return 0
    find "$root" -mindepth 1 -maxdepth 5 -type d -name 'checkpoint-*' \
        -exec test -f '{}/config.json' ';' \
        -exec test -f '{}/processor_config.json' ';' \
        -print | sort -V | tail -n 1
}

if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$(resolve_latest_checkpoint "$RUN_ROOT")"
fi

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" || ! -f "$MODEL_PATH/processor_config.json" ]]; then
    echo "[E3-A] could not resolve a usable checkpoint under: $RUN_ROOT" >&2
    echo "[E3-A] set MODEL_PATH to a checkpoint containing config.json and processor_config.json." >&2
    exit 1
fi

cd "$REPO_ROOT"
uv run python "${SCRIPT_DIR}/gr00t_inference_server.py" \
    --model-path "$MODEL_PATH" \
    --port 5000 \
    --device "$DEVICE" \
    --web-port "$WEB_PORT" \
    --prompt-version "$PROMPT_VERSION" \
    --target-area "$TARGET_AREA" \
    --action-step 1 \
    --sign-conditioning-mode "$SIGN_CONDITIONING_MODE" \
    --sign-ablation-mode "$SIGN_ABLATION_MODE" \
    --modality-config-path "${REPO_ROOT}/examples/SignNav/modality_config_signnav.py" \
    "$@"
