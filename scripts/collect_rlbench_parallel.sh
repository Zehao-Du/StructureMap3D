#!/bin/bash
set -euo pipefail

# Usage: ./collect_rlbench_parallel.sh [SAVE_ROOT]
# Example: ./collect_rlbench_parallel.sh /data3/hongyu/rlbench_saved_test

SAVE_ROOT=${1:-/data3/hongyu/rlbench_saved_test}

# Ensure CoppeliaSim env (adjust if installed elsewhere)
COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$HOME/CoppeliaSim}
export COPPELIASIM_ROOT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export MUJOCO_GL=egl

SCRIPT="/data3/hongyu/RLBench/rlbench/dataset_generator.py"

# Map tasks to GPUs (must be same length)
TASKS=(
  "close_box"
  "put_rubbish_in_bin"
  "close_laptop_lid"
  "water_plants"
  "unplug_charger"
  "toilet_seat_down"
)
GPUS=(2 4 5 6 8 9)

NUM_EPISODES=120
IMAGE_SIZE=224

mkdir -p "$SAVE_ROOT"
mkdir -p "$SAVE_ROOT/logs"

echo "Launching ${#TASKS[@]} collection jobs"
for idx in "${!TASKS[@]}"; do
  TASK=${TASKS[$idx]}
  GPU=${GPUS[$idx]}
  LOGDIR="$SAVE_ROOT/logs"
  LOGFILE="$LOGDIR/${TASK}.log"
  PIDFILE="$LOGDIR/${TASK}.pid"

  echo "Starting task=$TASK on GPU=$GPU -> log=$LOGFILE"

  CUDA_VISIBLE_DEVICES=$GPU xvfb-run -a --server-args='-screen 0 1280x1024x24' \
    python -u "$SCRIPT" \
      --save_path "$SAVE_ROOT" \
      --episodes_per_task $NUM_EPISODES \
      --processes 1 \
      --image_size $IMAGE_SIZE $IMAGE_SIZE \
      --renderer opengl \
      --arm_max_velocity 2.0 --arm_max_acceleration 8.0 \
      --tasks "$TASK" \
    > "$LOGFILE" 2>&1 &

  echo $! > "$PIDFILE"
  sleep 1
done

echo "All jobs launched. PID files in $SAVE_ROOT/logs"
echo "Monitor logs with: tail -f $SAVE_ROOT/logs/*.log"

# Wait for jobs to finish
for pidfile in "$SAVE_ROOT"/logs/*.pid; do
  pid=$(cat "$pidfile")
  if kill -0 "$pid" 2>/dev/null; then
    echo "Waiting for pid $pid"
    wait "$pid" || echo "Job $pid exited with non-zero status"
  fi
done

echo "All collection jobs finished."
