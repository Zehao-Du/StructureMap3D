import optuna
import subprocess
import os
import re

def objective(trial):
    # ======= 1. 定义 Policy 列表搜索空间 =======
    p_layers = trial.suggest_int("p_layers", 1, 4)
    p_hidden_dims = []
    for i in range(p_layers):
        w = trial.suggest_int(f"p_layer_{i}_width", 128, 1024, step=128)
        p_hidden_dims.append(w)
    
    policy_dims_str = str(p_hidden_dims).replace(" ", "")

    # ======= 2. 定义 GNN Encoder 参数 =======
    gnn_hidden_dim = trial.suggest_int("gnn_hidden_dim", 256, 768, step=128)
    gnn_layers = trial.suggest_int("gnn_layers", 4, 8, step=2)
    
    possible_heads = [4, 8, 12, 16, 24, 32]
    valid_heads = [h for h in possible_heads if gnn_hidden_dim % h == 0]
    gnn_heads = trial.suggest_categorical("gnn_heads", valid_heads)

    # ======= 3. 其他参数 =======
    policy_nonlinearty = trial.suggest_categorical("nonlin", ["relu", "tanh"])
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)

    # ======= 4. 构造命令 (修正路径) =======
    exp_id = f"trial_{trial.number}_PL{p_layers}_GH{gnn_hidden_dim}"
    
    # 基础路径前缀
    base = "agent.instantiate_config"

    cmd = [
        "python", "-m", "MapPolicy.search",
        "agent=Lift3d_Pointnet_GNN_BNMLP",
        
        # 修正后的 Policy 参数路径
        f"{base}.policy_hidden_dims={policy_dims_str}",
        f"{base}.policy_nonlinearty={policy_nonlinearty}",
        f"{base}.policy_dropout_rate={dropout}",
        
        # 修正后的 GNN Map Encoder 参数路径
        f"{base}.map_encoder.hidden_dim={gnn_hidden_dim}",
        f"{base}.map_encoder.num_layers={gnn_layers}",
        f"{base}.map_encoder.num_heads={gnn_heads}",
        
        "wandb.name=" + exp_id,
        "train.num_epochs=50",
    ]

    print(f"\n{'='*80}\n[Trial {trial.number}] Starting...\nCommand: {' '.join(cmd)}\n{'='*80}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1" # 指定显卡
    env["HYDRA_FULL_ERROR"] = "1"

    # ======= 5. 运行并捕获输出 =======
    best_val_loss = None
    
    # 使用 Popen 实时读取输出
    process = subprocess.Popen(
        cmd, 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )

    # 实时解析 stdout
    for line in process.stdout:
        print(line, end="") # 将子进程输出同步打印到主进程
        
        # 匹配 search.py 中的输出: "Search Trial Finished. Best Val Loss: 0.123456"
        if "Best Val Loss:" in line:
            match = re.search(r"Best Val Loss:\s+([0-9.]+)", line)
            if match:
                best_val_loss = float(match.group(1))

    process.wait()

    # ======= 6. 结果反馈 =======
    if process.returncode != 0:
        print(f"Trial {trial.number} failed with return code {process.returncode}")
        return float('inf') # 报错则返回无穷大

    if best_val_loss is None:
        print(f"Could not find Best Val Loss in output of Trial {trial.number}")
        return float('inf')

    return best_val_loss

if __name__ == "__main__":
    # 创建持久化的数据库(可选)，方便断点续传
    # storage = "sqlite:///optuna_search.db"
    
    study = optuna.create_study(
        # study_name="policy_architecture_search",
        # storage=storage,
        # load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5) # 5个epoch后效果太差就砍掉
    )
    
    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("Search interrupted by user.")

    print("\n" + "*"*50)
    print("BEST TRIAL COMPLETED")
    print(f"Best Val Loss: {study.best_value}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("*"*50)