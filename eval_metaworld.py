import os
# force EGL context to avoid DISPLAY errors when no X server
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('GLFW_CLIENT_API', 'EGL')
# unset DISPLAY so GLFW won't attempt X11
os.environ.pop('DISPLAY', None)

import sys
import argparse
import torch
import numpy as np
import wandb
import pathlib
from omegaconf import OmegaConf
from MapPolicy.helpers.graphics import PointCloud

# visualization support
import open3d as o3d
from hydra.utils import instantiate
from termcolor import colored
import tqdm

IMAGE_SIZE = 480

# monkey patch for gymnasium AutoresetMode missing attribute
try:
    import gymnasium.vector
    if not hasattr(gymnasium.vector, "AutoresetMode"):
        class _AutoresetModeStub:
            SAME_STEP = "same_step"
            NEXT_STEP = "next_step"
        gymnasium.vector.AutoresetMode = _AutoresetModeStub
except ImportError:
    pass

# ensure project root is on PYTHONPATH
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from MapPolicy.envs.metaworld_env import MetaWorldEnv
from MapPolicy.helpers.Common import set_seed, save_video_imageio
from MapPolicy.helpers.gymnasium import VideoWrapper

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate MetaWorld Policy from Local Output")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the local output directory (containing best_model.pth and wandb/)")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_video", action="store_true", default=True, help="Whether to save env videos")
    parser.add_argument("--video_dir", type=str, default="evaluation_videos", help="Directory to save env videos")
    parser.add_argument("--map_save_video", action="store_true", help="Whether to render and save map-construction videos")
    parser.add_argument("--map_video_dir", type=str, default="map_videos", help="Directory to save map videos")
    parser.add_argument("--save_first_ply", action="store_true", help="Write initial map point clouds of each episode to PLY")
    parser.add_argument("--first_ply_dir", type=str, default="first_pc", help="Directory for first-frame PLY files")
    parser.add_argument("--first_map_frames", type=int, default=5, help="Number of initial map frames to store per episode")
    parser.add_argument("--task_name", type=str, default=None, help="Override task name from config")
    parser.add_argument("--image_size", type=int, default=None, help="Override image size (affects renderer resolution)")
    return parser.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)

    # 1. Load Local Config
    output_path = pathlib.Path(args.output_dir)
    config_path = output_path / "wandb" / "latest-run" / "files" / "config.yaml"
    
    if not config_path.exists():
        # Try to find any config.yaml in wandb/run-*/files/
        import glob
        alt_configs = list(output_path.glob("wandb/run-*/files/config.yaml"))
        if alt_configs:
            config_path = alt_configs[-1]
            print(colored(f"latest-run not found, using {config_path}", "yellow"))
        else:
            print(colored(f"Config not found at {config_path}", "red"))
            return

    print(colored(f"Loading config from {config_path}...", "blue"))
    # WandB config.yaml has a wrapper format; use PyYAML to load raw data and
    # then unwrap any {value: ...} containers.  Avoid OmegaConf.resolve to skip
    # interpolation errors (wandb args often contain unresolvable hydra refs).
    import yaml

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    def _unwrap(obj):
        if isinstance(obj, dict):
            if "value" in obj and len(obj) == 1:
                return _unwrap(obj["value"])
            return {k: _unwrap(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_unwrap(v) for v in obj]
        else:
            return obj

    clean = _unwrap(full_config)
    config = OmegaConf.create(clean)
    
    # 2. Setup Environment
    task_name = args.task_name if args.task_name is not None else config.task_name
    print(colored(f"Initializing Environment: {task_name}", "green"))
    
    # Extract params from config
    max_episode_length = config.get("max_episode_length", 200)
    if "benchmark" in config and "evaluator_instantiate_config" in config.benchmark:
        max_episode_length = config.benchmark.evaluator_instantiate_config.get("max_episode_length", 200)
    
    image_size = config.get("image_size", IMAGE_SIZE)
    camera_name = config.get("camera_name", "corner")
    
    # Instantiate environment
    env = MetaWorldEnv(
        task_name=task_name,
        max_episode_length=max_episode_length,
        image_size=image_size,
        camera_name=camera_name,
        use_point_crop=config.get("use_point_crop", True),
        num_points=config.get("num_points", 1024),
        point_cloud_camera_names=config.get("point_cloud_camera_names", [camera_name])
    )
    env = VideoWrapper(env)

    # 3. Setup Model
    print(colored("Instantiating Model...", "green"))
    
    # Determine dims from environment
    obs_dict = env.reset()
    robot_state_dim = obs_dict["robot_state"].shape[-1]
    action_dim = 4 # MetaWorld default

    # Instantiate the agent model
    model = instantiate(
        config.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    )
    model = model.to(args.device)
    model.eval()

    # 4. Load Weights
    model_path = output_path / "best_model.pth"
    if not model_path.exists():
        print(colored(f"Model weights not found at {model_path}", "red"))
        return

    print(colored(f"Loading weights from {model_path}...", "green"))
    # load weights only (avoid future pickle-related warnings)
    state_dict = torch.load(model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)

    # prepare directories
    if args.save_video and not os.path.exists(args.video_dir):
        os.makedirs(args.video_dir)
    if args.map_save_video and not os.path.exists(args.map_video_dir):
        os.makedirs(args.map_video_dir)
    if args.save_first_ply and not os.path.exists(args.first_ply_dir):
        os.makedirs(args.first_ply_dir)

    success_count = 0
    total_reward = 0

    # offscreen renderer for structure maps (headless)
    map_renderer = None
    map_mat = None
    if args.map_save_video:
        # choose resolution equal to image_size (either config or override)
        render_size = args.image_size if args.image_size is not None else image_size
        map_renderer = o3d.visualization.rendering.OffscreenRenderer(render_size, render_size)
        map_mat = o3d.visualization.rendering.MaterialRecord()
        map_mat.shader = "defaultUnlit"

    def render_structure_map(structure_map, batch_idx=0):
        # turn structure_map into point cloud and render
        pts = structure_map.complete_point_cloud()[batch_idx].cpu().numpy()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        map_renderer.scene.clear_geometry()
        map_renderer.scene.add_geometry("map", pcd, map_mat)
        img = np.asarray(map_renderer.render_to_image())
        return img

    pbar = tqdm.tqdm(range(args.num_episodes), desc="Evaluating")
    for i in pbar:
        obs_dict = env.reset()
        done = False
        episode_reward = 0
        episode_success = False
        map_frames = []  # accumulate map-rendering frames per episode

        # optionally collect first few map clouds
        map_plys = []  # list of numpy arrays
        max_map = args.first_map_frames if args.save_first_ply else 0

        while not done:
            # Prepare inputs
            obs_img = obs_dict["image"]
            obs_point_cloud = obs_dict["point_cloud"]
            obs_point_cloud_no_robot = obs_dict["point_cloud_no_robot"]
            obs_robot_state = obs_dict["robot_state"]
            text = [env.text] # Environment description

            # To tensors
            img_tensor = torch.from_numpy(obs_img).float().permute(2, 0, 1).unsqueeze(0).to(args.device)
            pc_tensor = torch.from_numpy(obs_point_cloud).float().unsqueeze(0).to(args.device)
            pcnr_tensor = torch.from_numpy(obs_point_cloud_no_robot).float().unsqueeze(0).to(args.device)
            rs_tensor = torch.from_numpy(obs_robot_state).float().unsqueeze(0).to(args.device)

            with torch.no_grad():
                # Some models might use positional arguments or keyword arguments
                action = model(img_tensor, pc_tensor, pcnr_tensor, rs_tensor, text)
                # optionally render map
                if args.map_save_video or args.save_first_ply or True:
                    try:
                        # compute structure map
                        pcnr_normalized = PointCloud.normalize(pcnr_tensor)
                        structure_map = model.map_constructor(pcnr_normalized)
                        # debug: print predicted parameters
                        try:
                            # infer raw parameters by rerunning head
                            feats = model.map_constructor.point_cloud_encoder(pcnr_tensor)
                            params = model.map_constructor.estimation_head(feats)
                            sizes = params[:, : model.map_constructor.dims[0]]
                            positions = params[:, model.map_constructor.dims[0] : model.map_constructor.dims[1]]
                            rotations = params[:, model.map_constructor.dims[1] : model.map_constructor.dims[2]]
                            print(f"[map params] sizes {sizes.cpu().numpy()}\n positions {positions.cpu().numpy()}\n rotations {rotations.cpu().numpy()}")
                        except Exception:
                            pass
                        if args.map_save_video:
                            map_img = render_structure_map(structure_map, batch_idx=0)
                            map_frames.append(map_img)
                        if args.save_first_ply and len(map_plys) < max_map:
                            pcmap = structure_map.complete_point_cloud()[0].cpu().numpy()
                            map_plys.append(pcmap)
                    except Exception:
                        pass
            
            action = action.cpu().numpy().squeeze()
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_success = episode_success or info.get("success", False)

        if episode_success:
            success_count += 1
        total_reward += episode_reward

        # Save environment video
        if args.save_video:
            video_path = os.path.join(args.video_dir, f"eval_ep{i}_success{int(episode_success)}.mp4")
            env.save_video(video_path)

        # Save map video if requested
        if args.map_save_video and map_frames:
            map_path = os.path.join(args.map_video_dir, f"map_ep{i}_success{int(episode_success)}.mp4")
            save_video_imageio(np.stack(map_frames, axis=0), map_path, fps=30)
        # save first n map point clouds
        if args.save_first_ply and map_plys:
            import open3d as _o3d
            for idx, pcmap in enumerate(map_plys):
                pcd = _o3d.geometry.PointCloud(_o3d.utility.Vector3dVector(pcmap))
                _o3d.io.write_point_cloud(
                    os.path.join(args.first_ply_dir, f"ep{i}_map{idx}.ply"), pcd
                )

        pbar.set_postfix({
            "success_rate": f"{success_count / (i + 1):.2f}",
            "avg_reward": f"{total_reward / (i + 1):.2f}"
        })

    print(colored("\nEvaluation Finished!", "green"))
    print(f"Final Success Rate: {success_count / args.num_episodes:.2f}")
    print(f"Average Reward: {total_reward / args.num_episodes:.2f}")

if __name__ == "__main__":
    main()
