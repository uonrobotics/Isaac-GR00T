#!/usr/bin/env bash
set -e

unset PYTHONPATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source /opt/ros/jazzy/setup.bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/local_setup.bash

CMD_VEL_PORT="${CMD_VEL_PORT:-8766}"
if fuser -s "${CMD_VEL_PORT}/tcp"; then
    echo "[ROS2] stopping existing cmd_vel bridge on port ${CMD_VEL_PORT} ..."
    fuser -k "${CMD_VEL_PORT}/tcp"
fi

echo "[ROS2] starting minimal cmd_vel bridge ..."
/usr/bin/python3.12 "${REPO_ROOT}/examples/SignNav/inference_warehouse/ros2_cmdvel_bridge.py"
