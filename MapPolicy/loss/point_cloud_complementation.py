import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

# metaworld, camera extrinsic matrix
Metaworld_extrinsic_matrix = {
    "corner": np.array(
        [
            [-0.66173422, -0.48809537, 0.56909642, 0.0],
            [-0.31361979, 0.86966611, 0.38121317, 0.0],
            [0.68099225, -0.0737819, 0.7285642, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "corner2": np.array(
        [
            [0.56914086, -0.56424844, 0.59808225, 0.0],
            [0.23069754, 0.80774597, 0.54251738, 0.0],
            [-0.78921311, -0.17079271, 0.58989196, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}

# metaworld, camera intrinsic matrix
Metaworld_intrinsic_matrix = {
    "corner": np.array(
        [
            [270.3919, 0.0, 112.0],
            [0.0, 270.3919, 112.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "corner2": np.array(
        [
            [193.9897, 0.0, 112.0],
            [0.0, 193.9897, 112.0],
            [0.0, 0.0, 1.0],
        ]
    ),
}


def chamfer_loss(pc_a, pc_b):
    # pc_a: [B, N, 3], pc_b: [B, M, 3]
    # loss 直接返回平均距离，默认已经处理了双向
    loss, _ = chamfer_distance(pc_a[:, :, :3], pc_b[:, :, :3])
    return loss

def unidirectional_chamfer_loss(pc_a, pc_b):
    # point_reduction="mean" 对应外层的 .mean()
    # batch_reduction="mean" 对应 batch 维度的 .mean()
    loss, _ = chamfer_distance(
        pc_a[:, :, :3], 
        pc_b[:, :, :3], 
        single_directional=True # 设为 True 变为单向
    )
    return loss

def frust_masked_chamfer_loss(pc_gt, pc_pred, K, w2c, img_wh):
    """
    基于视锥裁剪的 Chamfer Loss
    
    参数:
        pc_pred: [B, N, 3] 全景预测点云 (世界坐标系)
        pc_gt:   [B, M, 3] 局部观测 GT 点云 (世界坐标系)
        K:       [3, 3] 相机内参矩阵
        w2c:     [4, 4] 世界坐标系到相机坐标系的变换矩阵
        img_wh:  tuple (width, height) 图像宽高
    """
    B, N, _ = pc_pred.shape
    W, H = img_wh
    device = pc_pred.device
    total_loss = 0.0

    # 遍历 Batch 处理（因为每张图在视野内的点数不同，无法直接并行 batch）
    for i in range(B):
        curr_pred = pc_pred[i]  # [N, 3]
        curr_gt = pc_gt[i][:, :3]      # [M, 3]
        curr_K = K              # [3, 3]
        curr_w2c = w2c          # [4, 4]

        # ---- 1. 将预测点云变换到相机坐标系 ----
        # 齐次变换
        curr_pred_homo = torch.cat([curr_pred[:, :3], torch.ones(N, 1, device=device)], dim=-1)
        pc_cam = (curr_pred_homo @ curr_w2c.T)[:, :3]  # [N, 3]

        # ---- 2. 投影并计算掩码 ----
        z = pc_cam[:, 2]
        # 投影 u = fx*x/z + cx, v = fy*y/z + cy
        pc_pixel = pc_cam @ curr_K.T
        u = pc_pixel[:, 0] / (z + 1e-6)
        v = pc_pixel[:, 1] / (z + 1e-6)

        # 视锥掩码：在图像范围内 且 深度在合理范围内 (0.1m - 50m)
        mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.1) & (z < 50.0)

        # ---- 3. 提取局部预测点云 ----
        pc_pred_local = curr_pred[mask]  # [N_local, 3]

        # ---- 4. 计算 Chamfer Loss ----
        if pc_pred_local.shape[0] > 10:  # 确保视野内有足够点，防止崩溃
            # chamfer_distance 期望输入 [B, P, 3]
            dist, _ = chamfer_distance(
                pc_pred_local.unsqueeze(0), 
                curr_gt.unsqueeze(0)
            )
            total_loss += dist
        else:
            # 如果视野内几乎没有预测点，但 GT 有点，施加一个简单的惩罚（可选）
            if curr_gt.shape[0] > 0:
                total_loss += curr_gt.abs().mean() * 0.01 

    return total_loss / B


def metaworld_frust_masked_chamfer_loss(pc_gt, pc_pred, camera_name, img_wh=[224, 224]):
    K = Metaworld_intrinsic_matrix[camera_name]
    w2c = Metaworld_extrinsic_matrix[camera_name]
    K = torch.from_numpy(K).float().to(pc_pred.device)
    w2c = torch.from_numpy(w2c).float().to(pc_pred.device)
    return frust_masked_chamfer_loss(pc_gt, pc_pred, K, w2c, img_wh)
