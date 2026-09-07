#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot/gr00t_n1d7-finetune+sim_v2_lerobot--20260824"
MODEL_PATH="${MODEL_PATH:-}"
DEVICE="${DEVICE:-cuda:0}"
WEB_PORT="${WEB_PORT:-9090}"
PROMPT_VERSION="${PROMPT_VERSION:-1}"
TARGET_AREA="${TARGET_AREA:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

resolve_latest_checkpoint() {
    local root="$1"
    find "$root" -maxdepth 1 -type d -name 'checkpoint-*' \
        -exec test -f '{}/config.json' ';' \
        -exec test -f '{}/processor_config.json' ';' \
        -print | sort -V | tail -n 1
}

if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$(resolve_latest_checkpoint "$RUN_ROOT")"
fi

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[E0] could not resolve a checkpoint under: $RUN_ROOT" >&2
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
    --modality-config-path "${REPO_ROOT}/examples/SignNav/modality_config_signnav.py" \
    "$@"
