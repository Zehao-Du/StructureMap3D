#!/bin/bash

export MUJOCO_GL=egl

# 2. 确保 Python 能找到你的项目包 (防止 ModuleNotFoundError)
# $(pwd) 代表当前目录，确保你在项目根目录运行此脚本
export PYTHONPATH=$PYTHONPATH:$(pwd)

TASKS=(
    "basketball"
    
    
    "lever-pull"
    
    "peg-unplug-side"
    
    "soccer"
    "stick-pull"
    
)

CAMERAS=(
    "corner"
    "corner2"
)

TOOL_SCRIPT="scripts/gen_data_metaworld.py"

# ================= Execution =================

echo "Starting data generation..."

for task in "${TASKS[@]}"; do
    for camera in "${CAMERAS[@]}"; do
        
        echo "------------------------------------------------"
        echo "Running :: Task: [$task] | Camera: [$camera]"
        echo "------------------------------------------------"

        # 执行 Python 命令
        # 注意: \ 符号用于换行，方便阅读
        python "$TOOL_SCRIPT" \
            --task-name "$task" \
            --camera-name "$camera" \
            --image-size 224 \
            --num-episodes 30 \
            --save-dir "data_new/metaworld" \
            --episode-length 200 \
            --quiet

        # 检查上一条命令是否成功
        if [ $? -ne 0 ]; then
            echo "❌ Error occurred in task: $task with camera: $camera"
            # 如果你希望出错就停止整个脚本，取消下面这一行的注释:
            exit 1
        else
            echo "✅ Finished successfully."
        fi
        
    done
done

echo "All tasks completed!"