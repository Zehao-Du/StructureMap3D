#!/bin/bash
set -euo pipefail

# Quick collection script: low-res, few episodes, fewer points per cloud
# Usage: ./collect_rlbench_fast.sh [SAVE_ROOT]

SAVE_ROOT=${1:-/data3/hongyu/rlbench_saved_test_fast}

COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$HOME/CoppeliaSim}
export COPPELIASIM_ROOT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export MUJOCO_GL=egl

SCRIPT="/data3/hongyu/RLBench/rlbench/dataset_generator.py"

# Tasks and GPUs (one-to-one)
TASKS=("close_box" "put_rubbish_in_bin" "close_laptop_lid" "water_plants" "unplug_charger" "toilet_seat_down")
GPUS=(2 4 5 6 8 9)

# Small settings for fast run
NUM_EPISODES=5
IMAGE_W=64
IMAGE_H=64
NUM_POINTS=256

mkdir -p "$SAVE_ROOT"
mkdir -p "$SAVE_ROOT/logs"

echo "Fast collection: $NUM_EPISODES episodes per task, image ${IMAGE_W}x${IMAGE_H}, points $NUM_POINTS"

for idx in "${!TASKS[@]}"; do
  TASK=${TASKS[$idx]}
  GPU=${GPUS[$idx]}
  LOGFILE="$SAVE_ROOT/logs/${TASK}.log"

  echo "Starting fast task=$TASK on GPU=$GPU -> $LOGFILE"

  CUDA_VISIBLE_DEVICES=$GPU xvfb-run -a --server-args='-screen 0 1024x768x24' \
    python -u "$SCRIPT" \
      --save_path "$SAVE_ROOT" \
      --episodes_per_task $NUM_EPISODES \
      --processes 1 \
      --image_size $IMAGE_W $IMAGE_H \
      --renderer opengl \
      --arm_max_velocity 2.0 --arm_max_acceleration 8.0 \
      --tasks "$TASK" \
    > "$LOGFILE" 2>&1 &

  sleep 0.5
done

echo "Launched fast collection jobs (logs in $SAVE_ROOT/logs)." 
echo "Monitor with: tail -f $SAVE_ROOT/logs/*.log & watch -n1 nvidia-smi"
