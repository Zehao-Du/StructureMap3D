import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import zarr

# Utility functions for visualizing episode-step data from zarr dataset
# Usage: python scripts/visualize_zarr_episode_step.py --zarr <path> --episode 0 --step 10 --point-cloud-key point_clouds_no_robot --output-dir <path>


def resolve_global_step(episode_ends: np.ndarray, episode_idx: int, step_idx: int) -> int:
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise IndexError(f"episode_idx={episode_idx} out of range [0, {len(episode_ends)-1}]")

    ep_end_exclusive = int(episode_ends[episode_idx])
    ep_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    ep_len = ep_end_exclusive - ep_start

    if step_idx < 0 or step_idx >= ep_len:
        raise IndexError(
            f"step_idx={step_idx} out of range [0, {ep_len-1}] for episode {episode_idx}"
        )

    return ep_start + step_idx


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def save_point_cloud_ply(point_cloud_xyzrgb: np.ndarray, save_path: pathlib.Path):
    pc = np.asarray(point_cloud_xyzrgb, dtype=np.float32)
    xyz = pc[:, :3]
    rgb = np.clip(pc[:, 3:6], 0, 255).astype(np.uint8)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize one episode-step image and point cloud from zarr")
    parser.add_argument("--zarr", type=str, required=True, help="Path to zarr dataset")
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument("--step", type=int, required=True, help="Step index inside episode")
    parser.add_argument(
        "--point-cloud-key",
        type=str,
        default="point_clouds",
        choices=["point_clouds", "point_clouds_no_robot"],
        help="Which point cloud key to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/zarr_vis",
        help="Directory to save image and ply files",
    )
    args = parser.parse_args()

    zarr_path = pathlib.Path(args.zarr)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]

    episode_ends = np.asarray(meta["episode_ends"][:])
    global_step = resolve_global_step(episode_ends, args.episode, args.step)

    image = np.asarray(data["images"][global_step])
    point_cloud = np.asarray(data[args.point_cloud_key][global_step])

    image_out = output_dir / f"ep{args.episode:03d}_step{args.step:03d}_image.png"
    pc_out = output_dir / f"ep{args.episode:03d}_step{args.step:03d}_{args.point_cloud_key}.ply"

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Image | ep={args.episode}, step={args.step}, global={global_step}")
    plt.tight_layout()
    plt.savefig(image_out, dpi=200)

    save_point_cloud_ply(point_cloud, pc_out)

    print(f"Saved image to: {image_out}")
    print(f"Saved point cloud ply to: {pc_out}")
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Episode length: {int(episode_ends[args.episode] - (0 if args.episode == 0 else episode_ends[args.episode-1]))}")
    plt.close("all")


if __name__ == "__main__":
    main()
