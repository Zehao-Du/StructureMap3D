#!/bin/bash

# ==========================================
# 1. CPU 优化设置
# ==========================================
NUMBER_THREAD=16
export OMP_NUM_THREADS=${NUMBER_THREAD}
export MKL_NUM_THREADS=${NUMBER_THREAD}
export OPENBLAS_NUM_THREADS=${NUMBER_THREAD}
export VECLIB_MAXIMUM_THREADS=${NUMBER_THREAD}
export NUMEXPR_NUM_THREADS=${NUMBER_THREAD}

PHYSICAL_DEVICE_ID=0

# ==========================================
# 2. 待运行的列表 (在此修改你想跑的任务组合)
# ==========================================
AGENTS=(
    "Lift3d_Pointnet_GNN"
    "Lift3d_Map_GNN"
    "Map_GNN"
    "Map_MLP_unidirectionalchamferloss"
    "Map_MLP_chamferloss"
)

TASKS=(
    "basketball"
    "bin-picking"
    "box-close"
    "coffee-pull"
    "coffee-push"
    "disassemble"
    "hammer"
    "hand-insert"
    "handle-pull"
    "handle-pull-side"
    "lever-pull"
    "peg-insert-side"
    "peg-unplug-side"
    "pick-out-of-hole"
    "pick-place"
    "pick-place-wall"
    "push"
    "push-back"
    "push-wall"
    "reach-wall"
    "shelf-place"
    "soccer"
    "stick-pull"
    "sweep"
    "sweep-into"
)

CAMERA_NAMES=(
    "corner"
    "corner2"
)

# Lambda 数值列表 (可以包含多个，如 1 2 5)
LAMBDA_VALS=(1)

# ==========================================
# 3. 核心执行逻辑
# ==========================================
echo "Starting unified training script..."

for AGENT_NAME in "${AGENTS[@]}"; do
    
    # --- [根据 Agent 自动绑定参数] ---
    case "${AGENT_NAME}" in
        "Lift3d_Map_GNN")
            BATCH_SIZE=24
            SKIP_EPOCHS=0
            NUM_EPOCHS=300
            LAMBDA_KEY="lambda_physical"
            ;;
        "Map_GNN")
            BATCH_SIZE=256
            SKIP_EPOCHS=0
            NUM_EPOCHS=300
            LAMBDA_KEY="lambda_physical"
            ;;
        "Map_MLP_unidirectionalchamferloss" | "Map_MLP_chamferloss")
            BATCH_SIZE=256
            SKIP_EPOCHS=0
            NUM_EPOCHS=200
            LAMBDA_KEY="lambda_map"
            ;;
        *)
            # 默认兜底配置
            BATCH_SIZE=256
            SKIP_EPOCHS=0
            NUM_EPOCHS=300
            LAMBDA_KEY="lambda_physical"
            echo "Warning: Agent ${AGENT_NAME} configuration not explicitly defined, using defaults."
            ;;
    esac

    for LAMBDA_VAL in "${LAMBDA_VALS[@]}"; do
        for TASK_NAME in "${TASKS[@]}"; do
            for CAMERA_NAME in "${CAMERA_NAMES[@]}"; do
                
                echo "=================================================="
                echo "Agent:  ${AGENT_NAME}"
                echo "Config: Batch=${BATCH_SIZE}, Epochs=${NUM_EPOCHS}, Skip=${SKIP_EPOCHS}"
                echo "Param:  ${LAMBDA_KEY}=${LAMBDA_VAL}"
                echo "Task:   ${TASK_NAME} | Camera: ${CAMERA_NAME}"
                echo "Device: cuda:${PHYSICAL_DEVICE_ID}"
                echo "=================================================="

                # --- 构造 Hydra 覆盖参数 ---
                AGENT_OVERRIDE="agent=${AGENT_NAME}"
                TASK_OVERRIDE="task_name=${TASK_NAME}"
                CAMERA_OVERRIDE="camera_name=${CAMERA_NAME}"
                LAMBDA_OVERRIDE="${LAMBDA_KEY}=${LAMBDA_VAL}"
                
                BATCH_OVERRIDE="dataloader.batch_size=${BATCH_SIZE}"
                SKIP_EPOCHS_OVERRIDE="evaluation.num_skip_epochs=${SKIP_EPOCHS}"
                EPOCHS_OVERRIDE="train.num_epochs=${NUM_EPOCHS}"
                
                # WandB 命名：包含 Agent 和任务信息
                WANDB_NAME_OVERRIDE="wandb.name=\${agent.name}_${TASK_NAME}_${CAMERA_NAME}"
                WANDB_NOTES_OVERRIDE="wandb.notes=${TASK_NAME}_${CAMERA_NAME}_${LAMBDA_KEY}_${LAMBDA_VAL}"

                # 输出目录：outputs/类型_数值/Agent名/任务_相机/时间戳
                HYDRA_DIR_OVERRIDE="hydra.run.dir=outputs/${LAMBDA_KEY}_${LAMBDA_VAL}/${AGENT_NAME}/${TASK_NAME}_${CAMERA_NAME}/${now:%Y-%m-%d_%H-%M-%S}"

                # 执行命令
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
                echo "Finished: ${AGENT_NAME} on ${TASK_NAME}_${CAMERA_NAME}"
                echo "--------------------------------------------------"
            done
        done
    done
done

echo "All training runs completed."