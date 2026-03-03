import torch

# original smart_loss_func rewritten to support weighted pos/rot/gripper components

def smart_loss_func(
    preds,
    actions,
    lambda_map=1.0,
    lambda_physical=1.0,
    lambda_pos=80.0,
    lambda_rot=40.0,
    lambda_gripper=1.0,
):
    """
    preds: 可能是 Tensor (action), 
           也可能是 Tuple (action, aux_loss_map, aux_loss_2)
           也可能是 Dict {"action": ..., "map_loss": ..., "aux_loss": ...}

    主动作假设为 7 维：
      - 0-2 维 position    (MSE)
      - 3-5 维 rotation    (MSE)
      - 6 维 gripper 开合 (-1/1)，使用二分类交叉熵

    lambda_map: 地图重构损失的权重
    lambda_physical: 第二个辅助损失的权重
    lambda_pos, lambda_rot, lambda_gripper: 主动作三部分的权重
    """

    # 统一提取预测和辅助loss
    if isinstance(preds, dict):
        pred_action = preds.get("action")
        aux_loss_1 = preds.get("map_loss", torch.tensor(0.0, device=pred_action.device))
        aux_loss_2 = preds.get("aux_loss", torch.tensor(0.0, device=pred_action.device))
    elif isinstance(preds, (tuple, list)):
        pred_action = preds[0]
        aux_loss_1 = preds[1] if len(preds) > 1 else torch.tensor(0.0, device=pred_action.device)
        aux_loss_2 = preds[2] if len(preds) > 2 else torch.tensor(0.0, device=pred_action.device)
    else:
        pred_action = preds
        aux_loss_1 = torch.tensor(0.0, device=pred_action.device)
        aux_loss_2 = torch.tensor(0.0, device=pred_action.device)

    # 主动作loss decomposed
    pos_loss = torch.nn.functional.mse_loss(pred_action[..., 0:3], actions[..., 0:3])
    rot_loss = torch.nn.functional.mse_loss(pred_action[..., 3:6], actions[..., 3:6])
    grip_pred = pred_action[..., 6:7]  # shape (B, 1)
    grip_bin = (actions[..., 6:7] > 0).float()
    grip_loss = torch.nn.functional.binary_cross_entropy_with_logits(grip_pred, grip_bin)

    main_loss = (
        lambda_pos * pos_loss
        + lambda_rot * rot_loss
        + lambda_gripper * grip_loss
    )

    total_loss = main_loss + (lambda_map * aux_loss_1) + (lambda_physical * aux_loss_2)

    return total_loss, {
        "loss/total": total_loss.item(),
        "loss/main_action": main_loss.item(),
        "loss/pos": pos_loss.item(),
        "loss/rot": rot_loss.item(),
        "loss/gripper": grip_loss.item(),
        "loss/aux_map_raw": aux_loss_1.item() if torch.is_tensor(aux_loss_1) else aux_loss_1,
        "loss/aux_physical_raw": aux_loss_2.item() if torch.is_tensor(aux_loss_2) else aux_loss_2,
        "lambda/map": lambda_map,
        "lambda/pos": lambda_pos,
        "lambda/rot": lambda_rot,
        "lambda/gripper": lambda_gripper,
    }

