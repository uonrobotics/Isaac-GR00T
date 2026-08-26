#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${SCRIPT_DIR}/setup_IsaacSim-ros_workspace.sh"

ENV_USD_PATH="${ENV_USD_PATH:-/nas/sujinkim/data/SignNav/_assets/warehouse/Industrial_NVD@10012/Industrial_NVD@10012/Assets/ArchVis/Industrial/Stages/IsaacWarehouse.usd}"
SPAWN_X="${SPAWN_X:-21.0}"
SPAWN_Y="${SPAWN_Y:-31.0}"
SPAWN_Z="${SPAWN_Z:-0.0}"
SPAWN_YAW="${SPAWN_YAW:--90.0}"
CAMERA_PRESET="${CAMERA_PRESET:-gemini_336l}"
IMAGE_FORMAT="${IMAGE_FORMAT:-jpeg}"
JPEG_QUALITY="${JPEG_QUALITY:-85}"
FLOOR_COLLISION_PRIM_PATH="${FLOOR_COLLISION_PRIM_PATH:-/World/Warehouse01/SM_Floor_A1}"
FLOOR_COLLISION_APPROXIMATION="${FLOOR_COLLISION_APPROXIMATION:-none}"
SIM_PORT="${SIM_PORT:-8765}"
INFERENCE_HOST="${INFERENCE_HOST:-127.0.0.1}"
INFERENCE_PORT="${INFERENCE_PORT:-5000}"
CMD_HOST="${CMD_HOST:-127.0.0.1}"
CMD_PORT="${CMD_PORT:-8766}"
CLIENT_HZ="${CLIENT_HZ:-5.0}"
ISAACSIM_EXTRA_ARGS="${ISAACSIM_EXTRA_ARGS:-}"

LOG_DIR="${REPO_ROOT}/.logs/signnav_inference_warehouse"
mkdir -p "$LOG_DIR"
ISAACSIM_LOG="${LOG_DIR}/isaacsim_min_$(date +%Y%m%d_%H%M%S).log"

echo "[MIN RUN] env_usd=${ENV_USD_PATH}"
echo "[MIN RUN] spawn=(${SPAWN_X}, ${SPAWN_Y}, ${SPAWN_Z}, yaw=${SPAWN_YAW})"
echo "[MIN RUN] camera=${CAMERA_PRESET}"
echo "[MIN RUN] camera_resolution=original image_format=${IMAGE_FORMAT} jpeg_quality=${JPEG_QUALITY}"
echo "[MIN RUN] floor_collision=${FLOOR_COLLISION_PRIM_PATH} approx=${FLOOR_COLLISION_APPROXIMATION}"
echo "[MIN RUN] isaacsim_log=${ISAACSIM_LOG}"

cleanup() {
    if [[ -n "${ISAACSIM_PID:-}" ]]; then
        kill "$ISAACSIM_PID" 2>/dev/null || true
        wait "$ISAACSIM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

ISAACSIM_EXTRA_ARGS_ARRAY=()
if [[ -n "$ISAACSIM_EXTRA_ARGS" ]]; then
    read -r -a ISAACSIM_EXTRA_ARGS_ARRAY <<< "$ISAACSIM_EXTRA_ARGS"
fi

PYTHONUNBUFFERED=1 /home/sujin/isaac-sim/python.sh \
    "${REPO_ROOT}/examples/SignNav/inference_warehouse/isaacsim_min_server.py" \
    --env-usd-path "$ENV_USD_PATH" \
    --spawn-x "$SPAWN_X" \
    --spawn-y "$SPAWN_Y" \
    --spawn-z "$SPAWN_Z" \
    --spawn-yaw "$SPAWN_YAW" \
    --camera-preset "$CAMERA_PRESET" \
    --image-format "$IMAGE_FORMAT" \
    --jpeg-quality "$JPEG_QUALITY" \
    --sim-port "$SIM_PORT" \
    --enable-ros2-bridge \
    --floor-collision-prim-path "$FLOOR_COLLISION_PRIM_PATH" \
    --floor-collision-approximation "$FLOOR_COLLISION_APPROXIMATION" \
    "${ISAACSIM_EXTRA_ARGS_ARRAY[@]}" >"$ISAACSIM_LOG" 2>&1 &
ISAACSIM_PID=$!

python3 "${REPO_ROOT}/examples/SignNav/inference_warehouse/gr00t_isaacsim_client.py" \
    --sim-host 127.0.0.1 \
    --sim-port "$SIM_PORT" \
    --inference-host "$INFERENCE_HOST" \
    --inference-port "$INFERENCE_PORT" \
    --cmd-host "$CMD_HOST" \
    --cmd-port "$CMD_PORT" \
    --hz "$CLIENT_HZ" \
    "$@"
