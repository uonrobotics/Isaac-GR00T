#!/usr/bin/env bash
set -e

MODEL_PATH=""
DEVICE="${DEVICE:-cuda:0}"
WEB_PORT="${WEB_PORT:-9090}"
PROMPT_VERSION="${PROMPT_VERSION:-2}"
TARGET_AREA="${TARGET_AREA:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
EXTRA_SERVER_ARGS=()

usage() {
    echo "Usage: ./3_start_gr00t_inference_server.sh /path/to/checkpoint [--device cuda:0] [--web-port 9090] [--prompt-version 1|2|3] [--target-area 1..12]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="${2:?missing value for --device}"
            shift 2
            ;;
        --web-port)
            WEB_PORT="${2:?missing value for --web-port}"
            shift 2
            ;;
        --prompt-version)
            PROMPT_VERSION="${2:?missing value for --prompt-version}"
            shift 2
            ;;
        --target-area)
            TARGET_AREA="${2:?missing value for --target-area}"
            shift 2
            ;;
        --*)
            EXTRA_SERVER_ARGS+=("$1")
            if [[ $# -gt 1 && "$2" != --* ]]; then
                EXTRA_SERVER_ARGS+=("$2")
                shift 2
            else
                shift
            fi
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -n "$MODEL_PATH" ]]; then
                echo "[GR00T] unexpected positional argument: $1" >&2
                usage
                exit 1
            fi
            MODEL_PATH="$1"
            shift
            ;;
    esac
done

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[GR00T] pass a checkpoint directory containing config.json" >&2
    usage
    exit 1
fi

cd "${REPO_ROOT}/examples/SignNav/inference_warehouse"
uv run python gr00t_inference_server.py \
    --model-path "$MODEL_PATH" \
    --port 5000 \
    --device "$DEVICE" \
    --web-port "$WEB_PORT" \
    --prompt-version "$PROMPT_VERSION" \
    --target-area "$TARGET_AREA" \
    --action-step 1 \
    --modality-config-path "${REPO_ROOT}/examples/SignNav/modality_config_signnav.py" \
    "${EXTRA_SERVER_ARGS[@]}"
