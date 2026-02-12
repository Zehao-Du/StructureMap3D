"""
将 RLBench 原始 demo 转为 MapPolicy 用的 zarr，保存到 data_new/rlbench。
需要先有 RLBench 原始数据（用 rlbench.dataset_generator 生成到 --rlbench-data-root）。

用法（在项目根目录）:
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  python scripts/gen_data_rlbench.py --task-name close_box --save-dir data_new/rlbench
"""
import argparse
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LIFT3D_ROOT = REPO_ROOT / "MapPolicy" / "models" / "LIFT3D"
RLBENCH_THIRD_PARTY = LIFT3D_ROOT / "third_party" / "RLBench"
SAVE_DIR_DEFAULT = REPO_ROOT / "data_new" / "rlbench"


def main():
    parser = argparse.ArgumentParser(
        description="Convert RLBench demos to zarr in data_new/rlbench"
    )
    parser.add_argument("--task-name", type=str, default="close_box")
    parser.add_argument("--rlbench-data-root", type=str, default="~/Data/RLBench_224")
    parser.add_argument("--camera-name", type=str, default="front")
    parser.add_argument("--save-dir", type=str, default=str(SAVE_DIR_DEFAULT))
    parser.add_argument("--num-episodes", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tool_path = LIFT3D_ROOT / "lift3d" / "tools" / "gen_data_rlbench.py"
    if not tool_path.exists():
        print(f"LIFT3D tool not found: {tool_path}")
        sys.exit(1)

    save_dir = pathlib.Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if RLBENCH_THIRD_PARTY.exists():
        env["PYTHONPATH"] = str(RLBENCH_THIRD_PARTY.resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(tool_path),
        "--rlbench-data-root",
        args.rlbench_data_root,
        "--task-name",
        args.task_name,
        "--camera-name",
        args.camera_name,
        "--point-cloud-camera-names",
        "front",
        "--rotation-representation",
        "quaternion",
        "--image-size",
        str(args.image_size),
        "--num-episodes",
        str(args.num_episodes),
        "--num-points",
        str(args.num_points),
        "--only-keypoints",
        "--save-dir",
        str(save_dir),
    ]
    if args.quiet:
        cmd.append("--quiet")

    print("Running:", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(LIFT3D_ROOT), env=env)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
