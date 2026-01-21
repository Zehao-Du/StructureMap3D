import zarr
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import argparse

def visualize_zarr_data(zarr_path, index=0):
    """
    可视化 Zarr 文件中的 RGB, Depth 和 Point Cloud 数据以验证正确性。
    """
    if not os.path.exists(zarr_path):
        print(f"Error: File not found at {zarr_path}")
        return

    print(f"Loading Zarr file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # 1. 读取数据
    # 注意：Zarr 读取出来的数据通常是只读的，copy() 一下变成可写的 numpy 数组，防止 Open3D 报错
    try:
        rgb_img = root['data/images'][index].copy()       # (H, W, 3)
        depth_img = root['data/depth'][index].copy()      # (H, W)
        pcd_data = root['data/point_clouds'][index].copy() # (H, W, 3) 或者 (N, 3)
    except IndexError:
        print(f"Error: Index {index} is out of bounds. Total items: {len(root['data/images'])}")
        return

    print(f"Data shapes - RGB: {rgb_img.shape}, Depth: {depth_img.shape}, PointCloud: {pcd_data.shape}")

    # ==========================================
    # 步骤 1: 2D 可视化 (检查 RGB 和 Depth 是否方向一致)
    # ==========================================
    plt.figure(figsize=(12, 5))
    
    # RGB
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title(f"Frame {index} - RGB Image")
    plt.axis('off')

    # Depth (使用伪彩色显示，更容易看出物体轮廓)
    plt.subplot(1, 2, 2)
    plt.imshow(depth_img, cmap='inferno')
    plt.title(f"Frame {index} - Depth Image\n(Check if upside down!)")
    plt.axis('off')
    
    print("\n--- 2D Check ---")
    print("Please check the Matplotlib window:")
    print("1. Is the RGB image upright?")
    print("2. Is the Depth image upright? (Look for the table/ground plane)")
    print("3. Do the edges of objects in RGB match the edges in Depth?")
    
    plt.show()

    # ==========================================
    # 步骤 2: 3D 可视化 (Open3D 检查点云和颜色映射)
    # ==========================================
    print("\n--- 3D Check ---")
    print("Preparing 3D Point Cloud visualization...")
    
    # 1. 展平形状为 (N, 3)
    points = pcd_data.reshape(-1, 3)
    colors = rgb_img.reshape(-1, 3) / 255.0

    # 2. 【核心修复】强制转换为 float64 并且 内存连续 (Contiguous)
    # Open3D 极度依赖内存连续性，reshape 后的数组往往不连续
    points = np.ascontiguousarray(points, dtype=np.float64)
    colors = np.ascontiguousarray(colors, dtype=np.float64)

    # (可选) 调试信息：检查一下是否有 NaN 或 Inf，这也会导致渲染问题
    if np.isnan(points).any() or np.isinf(points).any():
        print("Warning: Point cloud contains NaN or Inf values! Filtering them out for visualization...")
        # 过滤掉无效点
        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        print(f"Filtered {len(valid_mask) - len(points)} invalid points.")

    print(f"Final points shape: {points.shape}, dtype: {points.dtype}")
    print(f"Final colors shape: {colors.shape}, dtype: {colors.dtype}")

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # ... 后面的可视化代码保持不变 ...
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )

    print("Opening Open3D window...")
    o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                      window_name=f"Frame {index} - Point Cloud Verification",
                                      width=1024, height=768)

if __name__ == "__main__":
    # 解析命令行参数，方便你直接调用
    parser = argparse.ArgumentParser(description="Verify Zarr Data")
    # 默认路径设为你之前代码里生成的路径，你可以修改这里
    default_path = "/data2/zehao/3D_Map/data/metaworld/assembly_corner.zarr" 
    
    parser.add_argument("--path", type=str, default=default_path, help="Path to the .zarr file")
    parser.add_argument("--index", type=int, default=0, help="Index of the episode/frame to visualize")
    
    args = parser.parse_args()
    
    visualize_zarr_data(args.path, args.index)