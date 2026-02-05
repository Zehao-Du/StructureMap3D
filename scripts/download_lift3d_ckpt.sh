#!/bin/bash
# 在有网络的环境下运行此脚本，下载 LIFT3D 预训练权重到本地
# 下载完成后可将整个项目拷贝到无网络环境训练

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_DIR="$PROJECT_ROOT/MapPolicy/models/LIFT3D/lift3d/models/lift3d/ckpt"

mkdir -p "$CKPT_DIR"
cd "$PROJECT_ROOT"

echo "=== 下载 LIFT3D 预训练权重到 $CKPT_DIR ==="

BASE_URL="https://huggingface.co/jiayueru/Lift3d/resolve/main"

# 使用 wget 或 curl 下载（无需 Python 依赖）
for f in ViT-B-32.pt lift3d_clip_base.pth; do
    if [ -f "$CKPT_DIR/$f" ]; then
        echo "已存在，跳过: $f"
    elif command -v wget &>/dev/null; then
        echo "下载: $f ..."
        wget -q --show-progress -O "$CKPT_DIR/$f" "$BASE_URL/$f"
    elif command -v curl &>/dev/null; then
        echo "下载: $f ..."
        curl -L -o "$CKPT_DIR/$f" "$BASE_URL/$f"
    else
        echo "请手动下载并放置到 $CKPT_DIR/:"
        echo "  $BASE_URL/$f"
        exit 1
    fi
done

echo ""
echo "权重已保存到: $CKPT_DIR"
ls -lh "$CKPT_DIR"
echo ""
echo "可将此目录拷贝到无网络环境，训练时会自动使用本地文件。"
