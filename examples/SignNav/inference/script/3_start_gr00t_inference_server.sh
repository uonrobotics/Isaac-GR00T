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
    --modality-config-path "${REPO_ROOT}/examples/SignNav/modality_config_signnav.py"




# ./3_start_gr00t_inference_server.sh /nas/sujinkim/model/SignNav/gr00t/gr00t-finetune+sim_v1_lerobot_PROMPTv1/gr00t-finetune+sim_v1_lerobot_PROMPTv1--20260805/checkpoint-80000/ --prompt-version 1
# ./3_start_gr00t_inference_server.sh /nas/sujinkim/model/SignNav/gr00t/gr00t-finetune+sim_v1_lerobot_PROMPTv2/gr00t-finetune+sim_v1_lerobot_PROMPTv2--20260724/checkpoint-30000/ --prompt-version 2
# ./3_start_gr00t_inference_server.sh /nas/sujinkim/model/SignNav/gr00t/gr00t-finetune+sim_v1_lerobot_PROMPTv3/gr00t-finetune+sim_v1_lerobot_PROMPTv3--20260724/checkpoint-30000/ --prompt-version 3