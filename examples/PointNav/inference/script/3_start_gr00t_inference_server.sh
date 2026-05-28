#!/usr/bin/env bash
set -e

# DEFAULT_MODEL_PATH="/nas/sujinkim/model/goto/gr00t/gr00t-finetune+sim_v2_lerobot_multiview_gemini_336/gr00t-finetune+sim_v2_lerobot_multiview_gemini_336--20260522/checkpoint-60000"

DEFAULT_MODEL_PATH="/nas/sujinkim/model/goto/gr00t/gr00t-finetune+sim_v2_lerobot_single_gemini336l/gr00t-finetune+sim_v2_lerobot_single_gemini336l--20260522/checkpoint-60000"
# DEFAULT_MODEL_PATH="/nas/sujinkim/model/goto/gr00t/gr00t-finetune+sim_v2_lerobot_single_gemini345lg/gr00t-finetune+sim_v2_lerobot_single_gemini345lg--20260522/checkpoint-60000/"

MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"
DEVICE="${2:-cuda:0}"
WEB_PORT="${3:-9090}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[GR00T] model path does not exist: $MODEL_PATH" >&2
    echo "Usage: ./3_start_gr00t_inference_server.sh [/path/to/checkpoint] [device] [web_port]" >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[GR00T] invalid checkpoint: missing $MODEL_PATH/config.json" >&2
    echo "Pass the actual checkpoint directory, for example: .../checkpoint-60000" >&2
    exit 1
fi

cd /home/sujin/workspace/physical-ai/Isaac-GR00T/examples/PointNav/inference
uv run python gr00t_inference_server.py \
    --model-path "$MODEL_PATH" \
    --port 5000 \
    --device "$DEVICE" \
    --action-step 1 \
    --web-port "$WEB_PORT"
