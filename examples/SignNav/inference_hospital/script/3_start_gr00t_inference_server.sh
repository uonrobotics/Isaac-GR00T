#!/usr/bin/env bash
set -e

DEFAULT_MODEL_PATH=""

MODEL_PATH="$DEFAULT_MODEL_PATH"
DEVICE="cuda:0"
WEB_PORT="9090"
PROMPT_VERSION="2"
TARGET_AREA="1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
EXTRA_SERVER_ARGS=()

usage() {
    echo "Usage: ./3_start_gr00t_inference_server.sh /path/to/checkpoint [--device cuda:0] [--web-port 9090] [--prompt-version 1|2|3] [--target-area 1..12] [--enable-sam3-segmentation]" >&2
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
        --enable-sam3-segmentation|--disable-sam3-segmentation|--no-enable-sam3-segmentation)
            EXTRA_SERVER_ARGS+=("$1")
            shift
            ;;
        --segmented-view-key|--sam3-generator-path|--sam3-device|--sam3-prompt|--sam3-confidence|--sam3-min-mask-area|--sam3-checkpoint-path|--sam3-bpe-path|--sam3-output-kind|--sam3-merge-gap-ratio|--sam3-merge-gap-pixels|--sam3-bbox-padding-ratio|--sam3-bbox-padding-pixels|--sam3-bbox-line-thickness|--sam3-min-box-area-ratio|--sam3-min-box-width-ratio|--sam3-min-box-height-ratio)
            EXTRA_SERVER_ARGS+=("$1" "${2:?missing value for $1}")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "[GR00T] unknown option: $1" >&2
            usage
            exit 1
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

if [[ -z "$MODEL_PATH" ]]; then
    echo "[GR00T] model path is empty." >&2
    usage
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[GR00T] model path does not exist: $MODEL_PATH" >&2
    usage
    exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[GR00T] invalid checkpoint: missing $MODEL_PATH/config.json" >&2
    echo "Pass the actual checkpoint directory, for example: .../checkpoint-60000" >&2
    exit 1
fi

cd "${REPO_ROOT}/examples/SignNav/inference"
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




# ./3_start_gr00t_inference_server.sh /nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v1_lerobot_PROMPTv1/gr00t_n1d7-finetune+sim_v1_lerobot_PROMPTv1--20260803/checkpoint-80000/ --prompt-version 1
# ./3_start_gr00t_inference_server.sh /nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v1_lerobot_sign_bbox_overlay/gr00t_n1d7-finetune+sim_v1_lerobot_sign_bbox_overlay--20260805/checkpoint-80000/ --prompt-version 1 --enable-sam3-segmentation