#!/usr/bin/env bash
set -e

# conda 영향 제거
unset PYTHONPATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

source /opt/ros/jazzy/setup.bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/local_setup.bash

# ── Nav2 AMCL (Nova Carter 전용 launch) ───────────────────────────────────────
# Nova Carter AMCL launch. Isaac Sim publishes /clock, so Nav2/AMCL must use sim time.
echo "[ROS2] launching carter_navigation (AMCL) ..."
ros2 launch carter_navigation carter_navigation.launch.py &
NAV2_PID=$!

echo "[ROS2] waiting for AMCL to initialize (8s) ..."
sleep 8

# ── ROS2 bridge (AMCL 포즈 TCP 8767 + cmd_vel TCP 8766) ──────────────────────
echo "[ROS2] starting ros2_bridge ..."
/usr/bin/python3.12 /home/sujin/workspace/physical-ai/Isaac-GR00T/examples/PointNav/inference/ros2_amcl_cmdvel_bridge.py

# 종료 시 nav2도 같이 정리
kill $NAV2_PID 2>/dev/null || true
wait $NAV2_PID 2>/dev/null || true
echo "[ROS2] nav2 stopped."
