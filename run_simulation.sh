#!/usr/bin/env bash
# ==============================================================================
# Dex-TAG: Dexterous Task-Aware Grasp Synthesis & Manipulation Simulation
# Middleware: ROS 2 Jazzy Jalisco | Simulator: Gazebo Harmonic 8.15
# Platform: Ubuntu 24.04 LTS (WSL 2) | Hardware: NVIDIA GeForce RTX 5070 Ti
# ==============================================================================

set -e

echo "=========================================================================="
echo "  Dex-TAG: Dexterous Task-Aware Grasp Network & Manipulation Pipeline"
echo "  Repository: https://github.com/haree-bt/Dex-Tag-Model"
echo "=========================================================================="

# 1. Source ROS 2
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
    echo "[✓] Sourced ROS 2 Jazzy."
else
    echo "[✗] ERROR: /opt/ros/jazzy/setup.bash not found."
    exit 1
fi

# 2. Build / Source Workspace
WS_DIR="$HOME/ros2_ws_hare"
rm -f "$WS_DIR/src/tag_net_ros"
if [ ! -d "$WS_DIR/src/dex_tag_ros" ]; then
    echo "[!] Linking dex_tag_ros package into $WS_DIR/src..."
    ln -sfn /mnt/c/Users/hareh/.gemini/antigravity/scratch/dex_tag_net/dex_tag_ros "$WS_DIR/src/dex_tag_ros"
fi

cd "$WS_DIR"
colcon build --symlink-install --packages-select dex_tag_ros
source "$WS_DIR/install/setup.bash"
echo "[✓] dex_tag_ros package compiled and active."

# 3. Setup Gazebo Model & Resource Paths
MODELS_DIR="$WS_DIR/src/dex_tag_ros/models"
export GZ_SIM_RESOURCE_PATH="$MODELS_DIR:/opt/ros/jazzy/share:${GZ_SIM_RESOURCE_PATH:-}"
export SDF_PATH="$MODELS_DIR:/opt/ros/jazzy/share:${SDF_PATH:-}"

# 4. Launch Simulation
echo "=========================================================================="
echo ">>> Booting Gazebo Harmonic Simulator, 6-DoF UR5e Arm & Dex-TAG Node..."
echo "=========================================================================="

ros2 launch dex_tag_ros simulation.launch.py
