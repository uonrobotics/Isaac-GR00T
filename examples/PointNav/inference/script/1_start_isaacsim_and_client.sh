#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_IsaacSim-ros_workspace.sh"

# CAMERA_MODE="${1:-multiview}"
CAMERA_MODE="${1:-single}"

# CAMERA_PRESET="${2:-gemini_336}"
CAMERA_PRESET="${2:-gemini_336l}"
# CAMERA_PRESET="${2:-gemini_345lg}"

cleanup() {
    if [[ -n "${ISAACSIM_PID:-}" ]]; then
        kill "$ISAACSIM_PID" 2>/dev/null || true
        wait "$ISAACSIM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

/home/sujin/isaac-sim/python.sh /home/sujin/workspace/physical-ai/Isaac-GR00T/examples/PointNav/inference/isaacsim_server.py \
    --camera-mode "$CAMERA_MODE" \
    --camera-preset "$CAMERA_PRESET" \
    --enable-ros2-bridge &
ISAACSIM_PID=$!

python3 /home/sujin/workspace/physical-ai/Isaac-GR00T/examples/PointNav/inference/gr00t_isaacsim_client.py \
    --inference-host 127.0.0.1 \
    --inference-port 5000 \
    --sim-host 127.0.0.1 \
    --sim-port 8765 \
    --amcl-host 127.0.0.1 \
    --amcl-port 8767 \
    --cmd-host 127.0.0.1 \
    --cmd-port 8766
