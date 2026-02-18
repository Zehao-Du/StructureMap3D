import functools
import json
import os
import pathlib
import sys

import hydra
import torch
import wandb
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from termcolor import colored

from MapPolicy.helpers.Common import set_seed
from MapPolicy.helpers.Logger import Logger, WandBLogger
from MapPolicy.helpers.pytorch import AverageMeter, log_params_to_file

os.environ['HYDRA_FULL_ERROR'] = "1"
os.environ['MUJOCO_GL'] = 'egl'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

@hydra.main(version_base=None, config_path="config", config_name="train_metaworld")
def main(config) -> float: # 修改点：声明返回 float 供搜索算法使用
    #############################
    # log important information #
    #############################
    Logger.log_info(f'Running {colored("Search-Friendly Training", "cyan")}')
    Logger.log_info(f'Task: {config.task_name} | Agent: {config.agent.name}')

    set_seed(config.seed)

    ################
    # wandb logger #
    ################
    # 如果是自动化大规模搜索，建议将 mode 设置为 'disabled' 或者 'offline'
    wandb_logger = WandBLogger(
        config=config,
        hyperparameters=OmegaConf.to_container(config, resolve=True),
    )

    ##########################
    # datasets (No Evaluator) #
    ##########################
    train_dataset = instantiate(config.benchmark.dataset_instantiate_config, data_dir=config.dataset_dir, split="train")
    valid_dataset = instantiate(config.benchmark.dataset_instantiate_config, data_dir=config.dataset_dir, split="validation")

    ###############
    # dataloaders #
    ###############
    loader_kwargs = {
        "batch_size": config.dataloader.batch_size,
        "num_workers": config.dataloader.num_workers,
        "pin_memory": config.dataloader.pin_memory,
        "drop_last": config.dataloader.drop_last,
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

    # 获取维度信息
    _, _, _, sample_robot_state, _, sample_action, _ = next(iter(train_loader))
    robot_state_dim, action_dim = sample_robot_state.size(-1), sample_action.size(-1)

    #########
    # Model #
    #########
    model = instantiate(config.agent.instantiate_config, robot_state_dim=robot_state_dim, action_dim=action_dim)
    model = model.to(config.device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.train.learning_rate, weight_decay=1e-4)
    scheduler = instantiate(config.train.scheduler_instantiate_config, optimizer=optimizer)

    ###########################
    # Training Loop           #
    ###########################
    best_val_loss = float('inf')
    
    for cur_epoch in range(config.train.num_epochs):
        model.train()
        loss_train = AverageMeter()
        
        for cur_iter, (images, pcls, pcl_no_robot, robot_states, _, actions, texts) in enumerate(train_loader):
            images, pcls, pcl_no_robot = images.to(config.device), pcls.to(config.device), pcl_no_robot.to(config.device)
            robot_states, actions = robot_states.to(config.device), actions.to(config.device, non_blocking=True)

            preds = model(images, pcls, pcl_no_robot, robot_states, texts)
            loss_result = call(config.benchmark.loss_func, preds, actions)
            loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result

            optimizer.zero_grad()
            loss.backward()
            if config.train.clip_grad_value > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_value)
            optimizer.step()
            
            loss_train.update(loss.item())

        scheduler.step()

        # Validation Step
        model.eval()
        loss_val = AverageMeter()
        for images, pcls, pcl_no_robot, robot_states, _, actions, texts in valid_loader:
            images, pcls, pcl_no_robot = images.to(config.device), pcls.to(config.device), pcl_no_robot.to(config.device)
            robot_states, actions = robot_states.to(config.device), actions.to(config.device, non_blocking=True)
            
            with torch.no_grad():
                preds = model(images, pcls, pcl_no_robot, robot_states, texts)
                loss_result = call(config.benchmark.loss_func, preds, actions)
                v_loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
                loss_val.update(v_loss.item(), actions.shape[0])

        # 记录日志
        Logger.log_info(f"Epoch {cur_epoch}: Train Loss={loss_train.avg:.6f}, Val Loss={loss_val.avg:.6f}")
        wandb_logger.log({
            "epoch": cur_epoch,
            "train_epoch/loss": loss_train.avg,
            "validation/loss": loss_val.avg,
            "lr": scheduler.get_last_lr()[0]
        })

        # 更新最佳 Loss
        if loss_val.avg < best_val_loss:
            best_val_loss = loss_val.avg
            # 如果需要，可以在这里保存 checkpoint

    Logger.log_ok(f"Search Trial Finished. Best Val Loss: {best_val_loss:.6f}")
    
    # 必须要返回一个 float 值，Optuna 才能根据这个值进行最小化/最大化
    return best_val_loss


if __name__ == "__main__":
    main()