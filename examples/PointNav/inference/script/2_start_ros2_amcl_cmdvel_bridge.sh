#!/usr/bin/env bash
set -e

# conda 영향 제거
unset PYTHONPATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

source /opt/ros/jazzy/setup.bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/local_setup.bash

CARTER_SHARE="$HOME/IsaacSim-ros_workspaces/jazzy_ws/install/carter_navigation/share/carter_navigation"
MAP_FILE="${MAP_FILE:-$CARTER_SHARE/maps/sim_v2_env.yaml}"
PARAMS_FILE="${PARAMS_FILE:-$CARTER_SHARE/params/carter_navigation_params.yaml}"

cleanup() {
    kill ${BRIDGE_PID:-} ${AMCL_PID:-} ${SCAN_PID:-} 2>/dev/null || true
    wait ${BRIDGE_PID:-} ${AMCL_PID:-} ${SCAN_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── AMCL only ────────────────────────────────────────────────────────────────
# Full carter_navigation also starts controller/collision/docking nodes that
# publish to /cmd_vel. GR00T owns /cmd_vel here, so keep Nav2 to localization.
echo "[ROS2] launching pointcloud_to_laserscan for AMCL scan input ..."
ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node \
    --ros-args \
    -r cloud_in:=/front_3d_lidar/lidar_points \
    -r scan:=/scan \
    -p target_frame:=front_3d_lidar \
    -p transform_tolerance:=0.01 \
    -p min_height:=-0.4 \
    -p max_height:=1.5 \
    -p angle_min:=-1.5708 \
    -p angle_max:=1.5708 \
    -p angle_increment:=0.0087 \
    -p scan_time:=0.3333 \
    -p range_min:=0.05 \
    -p range_max:=100.0 \
    -p use_inf:=true \
    -p inf_epsilon:=1.0 \
    -p use_sim_time:=true &
SCAN_PID=$!

echo "[ROS2] launching Nav2 localization only (map_server + AMCL) ..."
ros2 launch nav2_bringup localization_launch.py \
    map:="$MAP_FILE" \
    params_file:="$PARAMS_FILE" \
    use_sim_time:=true &
AMCL_PID=$!

echo "[ROS2] waiting for AMCL/localization to initialize (8s) ..."
sleep 8

# ── ROS2 bridge (AMCL 포즈 TCP 8767 + cmd_vel TCP 8766) ──────────────────────
echo "[ROS2] starting ros2_bridge ..."
/usr/bin/python3.12 /home/sujin/workspace/physical-ai/Isaac-GR00T/examples/PointNav/inference/ros2_amcl_cmdvel_bridge.py &
BRIDGE_PID=$!
wait $BRIDGE_PID
