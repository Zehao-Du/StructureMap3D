#!/bin/bash
# --- CPU 优化设置 ---
NUMBER_THREAD=16
export OMP_NUM_THREADS=${NUMBER_THREAD}
export MKL_NUM_THREADS=${NUMBER_THREAD}
export OPENBLAS_NUM_THREADS=${NUMBER_THREAD}
export VECLIB_MAXIMUM_THREADS=${NUMBER_THREAD}
export NUMEXPR_NUM_THREADS=${NUMBER_THREAD}
PHYSICAL_DEVICE_ID=7
BATCH_SIZE=24
NUM_SKIP_EPOCHS=0
NUM_EPOCHS=300
# LAMBDA_PHYSICALS=(0.1 0.5 1 2 5 10 20)
LAMBDA_PHYSICALS=(1)
AGENTS=(
    "Lift3d_Map_GNN"
    # "Map_MLP_chamferloss"
    # "Map_GNN"
)

TASK_CAMERA_PAIRS=(
    "bin-picking:corner"
    "bin-picking:corner2"
    # "box-close:corner"
    "box-close:corner2"
)
echo "Starting training runs with overridden parameters..."
for LAMBDA_VAL in "${LAMBDA_PHYSICALS[@]}"; do
    for AGENT_NAME in "${AGENTS[@]}"; do
        for TASK_CAMERA_PAIR in "${TASK_CAMERA_PAIRS[@]}"; do
            # 分割任务和相机视角
            IFS=':' read -r TASK_NAME CAMERA_NAME <<< "$TASK_CAMERA_PAIR"
            
            echo "=================================================="
            echo "Lambda: ${LAMBDA_VAL}"
            echo "Agent:  ${AGENT_NAME}"
            echo "Task:   ${TASK_NAME}"
            echo "Camera: ${CAMERA_NAME}"
            echo "Batch:  ${BATCH_SIZE}"
            echo "Skip Epochs: ${NUM_SKIP_EPOCHS}"
            echo "Device: cuda: ${PHYSICAL_DEVICE_ID}"
            echo "=================================================="
            # --- 配置覆盖参数 ---
            # 覆盖 defaults 里的 agent 配置
            AGENT_OVERRIDE="agent=${AGENT_NAME}"
            # 覆盖基础参数
            TASK_OVERRIDE="task_name=${TASK_NAME}"
            CAMERA_OVERRIDE="camera_name=${CAMERA_NAME}"
            LAMBDA_OVERRIDE="lambda_physical=${LAMBDA_VAL}"
            # 覆盖 Batch Size
            BATCH_OVERRIDE="dataloader.batch_size=${BATCH_SIZE}"
            # [新增] 覆盖 evaluation.num_skip_epochs
            SKIP_EPOCHS_OVERRIDE="evaluation.num_skip_epochs=${NUM_SKIP_EPOCHS}"
            EPOCHS_OVERRIDE="train.num_epochs=${NUM_EPOCHS}"
            
            # 构造 wandb 覆盖
            WANDB_NAME_OVERRIDE="wandb.name=\${agent.name}_${TASK_NAME}_${CAMERA_NAME}"
            WANDB_NOTES_OVERRIDE="wandb.notes=${TASK_NAME}_${CAMERA_NAME}"
            HYDRA_DIR_OVERRIDE="hydra.run.dir=outputs/lambda_physical_${LAMBDA_VAL}/${AGENT_NAME}/${TASK_NAME}_${CAMERA_NAME}/${now:%Y-%m-%d_%H-%M-%S}"
            # 执行训练命令
            CUDA_VISIBLE_DEVICES=${PHYSICAL_DEVICE_ID} python -m MapPolicy.train \
                "${AGENT_OVERRIDE}" \
                "${TASK_OVERRIDE}" \
                "${CAMERA_OVERRIDE}" \
                "${LAMBDA_OVERRIDE}" \
                "${BATCH_OVERRIDE}" \
                "${SKIP_EPOCHS_OVERRIDE}" \
                "${EPOCHS_OVERRIDE}" \
                "${WANDB_NAME_OVERRIDE}" \
                "${WANDB_NOTES_OVERRIDE}" \
                "${HYDRA_DIR_OVERRIDE}"
                
            echo "--------------------------------------------------"
            echo "Finished: Lambda ${LAMBDA_VAL} for ${AGENT_NAME} on ${TASK_NAME}_${CAMERA_NAME}"
            echo "--------------------------------------------------"
        done
    done
done
echo "All training runs completed."