from pytorch3d.loss import chamfer_distance

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