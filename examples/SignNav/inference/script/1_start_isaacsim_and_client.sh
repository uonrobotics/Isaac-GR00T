#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${SCRIPT_DIR}/setup_IsaacSim-ros_workspace.sh"

# CAMERA_SETUP=(single gemini_336 default)
# CAMERA_SETUP=(single gemini_336l default)
# CAMERA_SETUP=(single gemini_345lg default)
CAMERA_SETUP=(single gemini_336l default)

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

/home/sujin/isaac-sim/python.sh "${REPO_ROOT}/examples/SignNav/inference/isaacsim_server.py" \
    --camera-mode "$CAMERA_MODE" \
    --camera-preset "$CAMERA_PRESET" \
    --camera-layout "$CAMERA_LAYOUT" \
    --viewport-renderer rtx \
    --camera-renderer realtime \
    --enable-ros2-bridge &
ISAACSIM_PID=$!

python3 "${REPO_ROOT}/examples/SignNav/inference/gr00t_isaacsim_client.py" \
    --inference-host 127.0.0.1 \
    --inference-port 5000 \
    --sim-host 127.0.0.1 \
    --sim-port 8765 \
    --cmd-host 127.0.0.1 \
    --cmd-port 8766 \
    "$@"
