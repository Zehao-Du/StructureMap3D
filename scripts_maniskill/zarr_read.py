import zarr
import numpy as np

# 1. 加载Zarr数据集（支持本地文件路径/文件夹，或云存储路径）
# 方式1：加载整个Zarr存储（文件夹形式）
zarr_store = zarr.open("/data2/lirui/StructureMap3D/data_new/maniskill/StackCube-v1_base_camera.zarr", mode="r")  # r=只读，避免修改数据

# 2. 查看数据结构（关键：确认维度、形状、数据类型）
print("=== Zarr 数据结构 ===")

# 递归打印组中的内容

def print_zarr_tree(node, indent=0):
    prefix = "  " * indent
    if isinstance(node, zarr.Group):
        print(f"{prefix}<Group> {list(node.name.split('/'))[-1] or '/'}")
        for key in node.keys():
            print_zarr_tree(node[key], indent + 1)
    elif isinstance(node, zarr.Array):
        print(f"{prefix}<Array> {list(node.name.split('/'))[-1]}  shape={node.shape}  dtype={node.dtype}")
    else:
        print(f"{prefix}<Unknown> {node}")

print_zarr_tree(zarr_store)

def save_points_to_ply(points: np.ndarray, ply_path: str):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"点云形状不合法，期望[N, >=3]，实际为 {points.shape}")

    xyz = points[:, :3].astype(np.float32)
    rgb = None
    if points.shape[1] >= 6:
        rgb_raw = points[:, 3:6].astype(np.float32)
        if rgb_raw.size > 0 and np.nanmax(rgb_raw) <= 1.0:
            rgb_raw = rgb_raw * 255.0
        rgb = np.clip(np.nan_to_num(rgb_raw, nan=0.0), 0, 255).astype(np.uint8)

    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for x, y, z in xyz:
                f.write(f"{x} {y} {z}\n")
        else:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


# 读取 data 组并保存 point_clouds / point_clouds_no_robot 的第一帧为 PLY
if isinstance(zarr_store, zarr.Group) and "data" in zarr_store and isinstance(zarr_store["data"], zarr.Group):
    data_group = zarr_store["data"]

    targets = {
        "point_clouds": "first_frame_point_cloud_with_robot.ply",
        "point_clouds_no_robot": "first_frame_point_cloud_no_robot.ply",
    }

    for key, out_name in targets.items():
        if key not in data_group:
            print(f"[跳过] data 组中不存在键: {key}")
            continue

        arr = data_group[key]
        if not isinstance(arr, zarr.Array):
            print(f"[跳过] {key} 不是数组，类型为: {type(arr)}")
            continue

        print(f"\n数组 '{key}' 信息:")
        print("  shape:", arr.shape)
        print("  dtype:", arr.dtype)
        print("  dims:", arr.attrs.get("dimensions", "无"))

        if arr.shape[0] == 0:
            print(f"[跳过] {key} 为空数组，无法导出第一帧")
            continue

        first_frame = arr[0]
        save_points_to_ply(first_frame, out_name)
        print(f"[保存] {key} 第一帧 -> {out_name}")
else:
    print("\n未找到 data 组，无法导出 point_clouds / point_clouds_no_robot 的第一帧。")
