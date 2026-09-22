#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260909--w1_5_0p5--signw0p05/checkpoint-100000/"
MODEL_PATH="${MODEL_PATH:-$RUN_ROOT}"
DEVICE="${DEVICE:-cuda:0}"
WEB_PORT="${WEB_PORT:-9090}"
PROMPT_VERSION="${PROMPT_VERSION:-1}"
TARGET_AREA="${TARGET_AREA:-1}"
# SIGN_CONDITIONING_MODE="${SIGN_CONDITIONING_MODE:-pred}"
SIGN_CONDITIONING_MODE="${SIGN_CONDITIONING_MODE:-gt_bbox_status}"

# Ablation selection: leave exactly one B/C/D line uncommented, or leave all
# three commented for A (normal). An exported SIGN_ABLATION_MODE also works.
# SIGN_ABLATION_MODE="${SIGN_ABLATION_MODE:-normal}"  # A: sign_hidden + bbox
SIGN_ABLATION_MODE="no_grounded_token"           # B: append no grounded token
# SIGN_ABLATION_MODE="hidden_only"                 # C: bbox feature = 0
# SIGN_ABLATION_MODE="bbox_only"                   # D: sign feature = 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

resolve_latest_checkpoint() {
    local root="$1"
    find "$root" -mindepth 2 -maxdepth 4 -type d -name 'checkpoint-*' \
        -exec test -f '{}/config.json' ';' \
        -exec test -f '{}/processor_config.json' ';' \
        -print | sort -V | tail -n 1
}

if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$(resolve_latest_checkpoint "$RUN_ROOT")"
fi

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[E1-C] could not resolve a checkpoint under: $RUN_ROOT" >&2
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
