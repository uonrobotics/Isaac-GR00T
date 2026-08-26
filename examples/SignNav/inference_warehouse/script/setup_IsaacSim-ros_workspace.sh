#!/usr/bin/env bash

set -e

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    source /opt/ros/jazzy/setup.bash
fi

if [[ -f "${HOME}/IsaacSim-ros_workspaces/jazzy_ws/install/local_setup.bash" ]]; then
    source "${HOME}/IsaacSim-ros_workspaces/jazzy_ws/install/local_setup.bash"
fi
