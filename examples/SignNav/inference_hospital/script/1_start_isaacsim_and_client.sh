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

ENV_USD_PATH="${ENV_USD_PATH:-/nas/sujinkim/data/SignNav/_assets/hospital/hospital_with_signs_and_waypoints.usd}"
OCCUPANCY_MAP_PNG_PATH="${OCCUPANCY_MAP_PNG_PATH:-/nas/sujinkim/data/SignNav/_assets/hospital/hospital_occupancy_map.png}"
OCCUPANCY_MAP_YAML_PATH="${OCCUPANCY_MAP_YAML_PATH:-}"
SPAWN_MODE="${SPAWN_MODE:-current}"
# SPAWN_MODE="${SPAWN_MODE:-episode_action_start}"
SPAWN_Z="${SPAWN_Z:-0.0}"
USE_RECORDED_SPAWN_Z="${USE_RECORDED_SPAWN_Z:-0}"
ADD_FLOOR_COLLISION="${ADD_FLOOR_COLLISION:-1}"
FLOOR_COLLISION_PRIM_PATHS="${FLOOR_COLLISION_PRIM_PATHS:-/World/Warehouse01/SM_Floor_A1}"
FLOOR_COLLISION_AUTO_KEYWORDS="${FLOOR_COLLISION_AUTO_KEYWORDS:-}"
FLOOR_COLLISION_APPROXIMATION="${FLOOR_COLLISION_APPROXIMATION:-none}"
ADD_FALLBACK_GROUND="${ADD_FALLBACK_GROUND:-0}"
FALLBACK_GROUND_Z="${FALLBACK_GROUND_Z:-0.0}"
FALLBACK_GROUND_SIZE="${FALLBACK_GROUND_SIZE:-200.0}"

ACTION_TRAJECTORY_ROOT="${ACTION_TRAJECTORY_ROOT:-/nas/sujinkim/data/SignNav/sim_v2/action}"
SPAWN_WAYPOINTS_PATH="${SPAWN_WAYPOINTS_PATH:-/nas/sujinkim/data/SignNav/_assets/hospital/hospital_waypoint_graphs/waypoints.npy}"

ISAACSIM_EXTRA_ARGS="${ISAACSIM_EXTRA_ARGS:-}"
LOG_DIR="${REPO_ROOT}/.logs/signnav"
mkdir -p "$LOG_DIR"
ISAACSIM_LOG="${LOG_DIR}/isaacsim_$(date +%Y%m%d_%H%M%S).log"

echo "[CAMERA] mode=${CAMERA_MODE} preset=${CAMERA_PRESET} layout=${CAMERA_LAYOUT}"
echo "[WORLD] env_usd=${ENV_USD_PATH}"
echo "[WORLD] occupancy_png=${OCCUPANCY_MAP_PNG_PATH} occupancy_yaml=${OCCUPANCY_MAP_YAML_PATH}"
echo "[WORLD] floor_collision=${ADD_FLOOR_COLLISION} paths=${FLOOR_COLLISION_PRIM_PATHS} keywords=${FLOOR_COLLISION_AUTO_KEYWORDS} approx=${FLOOR_COLLISION_APPROXIMATION}"
echo "[WORLD] fallback_ground=${ADD_FALLBACK_GROUND} z=${FALLBACK_GROUND_Z} size=${FALLBACK_GROUND_SIZE}"
echo "[SPAWN] mode=${SPAWN_MODE} z=${SPAWN_Z} use_recorded_z=${USE_RECORDED_SPAWN_Z} action_root=${ACTION_TRAJECTORY_ROOT} waypoints=${SPAWN_WAYPOINTS_PATH}"
echo "[ISAACSIM] log: ${ISAACSIM_LOG}"

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

FALLBACK_GROUND_ARGS=()
if [[ "$ADD_FALLBACK_GROUND" == "1" || "$ADD_FALLBACK_GROUND" == "true" ]]; then
    FALLBACK_GROUND_ARGS=(
        --add-fallback-ground
        --fallback-ground-z "$FALLBACK_GROUND_Z"
        --fallback-ground-size "$FALLBACK_GROUND_SIZE"
    )
fi

FLOOR_COLLISION_ARGS=()
if [[ "$ADD_FLOOR_COLLISION" == "1" || "$ADD_FLOOR_COLLISION" == "true" ]]; then
    FLOOR_COLLISION_ARGS=(
        --add-floor-collision
        --floor-collision-prim-paths "$FLOOR_COLLISION_PRIM_PATHS"
        --floor-collision-auto-keywords "$FLOOR_COLLISION_AUTO_KEYWORDS"
        --floor-collision-approximation "$FLOOR_COLLISION_APPROXIMATION"
    )
fi

RECORDED_SPAWN_Z_ARGS=()
if [[ "$USE_RECORDED_SPAWN_Z" == "1" || "$USE_RECORDED_SPAWN_Z" == "true" ]]; then
    RECORDED_SPAWN_Z_ARGS=(--use-recorded-spawn-z)
fi

PYTHONUNBUFFERED=1 /home/sujin/isaac-sim/python.sh "${REPO_ROOT}/examples/SignNav/inference/isaacsim_server.py" \
    --camera-mode "$CAMERA_MODE" \
    --camera-preset "$CAMERA_PRESET" \
    --camera-layout "$CAMERA_LAYOUT" \
    --viewport-renderer rtx \
    --camera-renderer realtime \
    --enable-ros2-bridge \
    --env-usd-path "$ENV_USD_PATH" \
    --occupancy-map-png-path "$OCCUPANCY_MAP_PNG_PATH" \
    --occupancy-map-yaml-path "$OCCUPANCY_MAP_YAML_PATH" \
    --spawn-mode "$SPAWN_MODE" \
    --action-trajectory-root "$ACTION_TRAJECTORY_ROOT" \
    --spawn-waypoints-path "$SPAWN_WAYPOINTS_PATH" \
    --spawn-z "$SPAWN_Z" \
    "${RECORDED_SPAWN_Z_ARGS[@]}" \
    "${FLOOR_COLLISION_ARGS[@]}" \
    "${FALLBACK_GROUND_ARGS[@]}" \
    "${ISAACSIM_EXTRA_ARGS_ARRAY[@]}" >"$ISAACSIM_LOG" 2>&1 &
ISAACSIM_PID=$!

python3 "${REPO_ROOT}/examples/SignNav/inference/gr00t_isaacsim_client.py" \
    --inference-host 127.0.0.1 \
    --inference-port 5000 \
    --sim-host 127.0.0.1 \
    --sim-port 8765 \
    --cmd-host 127.0.0.1 \
    --cmd-port 8766 \
    "$@"
