#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_IsaacSim-ros_workspace.sh"

# CAMERA_SETUP=(single gemini_336 default)
# CAMERA_SETUP=(single gemini_336l default)
# CAMERA_SETUP=(single gemini_345lg default)
# CAMERA_SETUP=(multiview gemini_336 default)
# CAMERA_SETUP=(multiview gemini_336l gemini336l_driveway_view)
# CAMERA_SETUP=(multiview gemini_336l gemini336l_high_dual_view) # high-dual-landscape
# CAMERA_SETUP=(multiview gemini_336l_portrait gemini336l_high_dual_view) # high-dual-portrait
CAMERA_SETUP=(multiview gemini_336l_portrait gemini336l_high_dual_view_concat) # high-dual-portrait-concat

CAMERA_MODE="${CAMERA_SETUP[0]}"
CAMERA_PRESET="${CAMERA_SETUP[1]}"
CAMERA_LAYOUT="${CAMERA_SETUP[2]}"

echo "[CAMERA] mode=${CAMERA_MODE} preset=${CAMERA_PRESET} layout=${CAMERA_LAYOUT}"

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
    --camera-layout "$CAMERA_LAYOUT" \
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
    --cmd-port 8766 \
    --no-sim-pose-fallback \
    "$@"
