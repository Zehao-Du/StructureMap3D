
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 原始 demo 保存路径（第一步生成到这里）
SOURCE_ROOT="${SOURCE_ROOT:-$HOME/Data/RLBench_224}"
# 最终 zarr 保存路径（第二步输出到这里）
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/data_new/rlbench}"
EPISODES="${EPISODES:-120}"

echo "Repo root:    $REPO_ROOT"
echo "Source root:  $SOURCE_ROOT (step 1)"
echo "Save dir:     $SAVE_DIR (step 2)"
echo "Episodes:     $EPISODES (headless, 无可视化)"

mkdir -p "$SOURCE_ROOT"
mkdir -p "$SAVE_DIR"

# 无可视化：让 Qt 在 xvfb 下用 xcb，避免 "Could not find the Qt platform plugin xcb"
if [ -n "$DISPLAY" ]; then
  export QT_QPA_PLATFORM=xcb
fi

# Step 1: 用 RLBench 生成 close_box 原始 demo（headless，不弹窗）
echo ""
echo "========== Step 1: 生成 RLBench 原始 demo (close_box, headless) =========="
python -m rlbench.dataset_generator \
    --save_path "$SOURCE_ROOT" \
    --image_size 224 224 \
    --tasks close_box \
    --episodes_per_task "$EPISODES"

# Step 2: 转为 zarr 到 data_new/rlbench
echo ""
echo "========== Step 2: 转为 zarr 到 data_new/rlbench =========="
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT"
python scripts/gen_data_rlbench.py \
    --task-name close_box \
    --rlbench-data-root "$SOURCE_ROOT" \
    --save-dir "$SAVE_DIR" \
    --num-episodes "$EPISODES"

echo ""
echo "Done. Zarr 已写入: $SAVE_DIR/close_box.zarr"
echo "训练时 dataset_dir 已配置为: \${project_root}/data_new/rlbench/\${task_name}.zarr"
