import os
import sys

# ensure project root is on PYTHONPATH so this file can be run directly
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import argparse

import torch

from MapPolicy.envs.maniskill_wrapper_env import ManiSkillEvaluator


class RandomPolicy(torch.nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.action_dim = action_dim

    def forward(self, images, point_clouds, point_cloud_no_robot, robot_states, texts):
        batch_size = images.shape[0]
        return torch.rand(batch_size, self.action_dim, device=images.device) * 2 - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=str, default="PickCube-v1")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--camera-name", type=str, default="base_camera")
    parser.add_argument("--obs-mode", type=str, default="pointcloud")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    evaluator = ManiSkillEvaluator(
        task_id=args.task_id,
        max_episode_length=args.episode_length,
        image_size=args.image_size,
        camera_name=args.camera_name,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        num_points=args.num_points,
    )

    action_dim = evaluator.env.action_space.shape[-1]
    policy = RandomPolicy(action_dim).to(args.device)

    success, reward = evaluator.evaluate(args.episodes, policy, verbose=args.verbose)
    print({"success": float(success), "reward": float(reward)})

    evaluator.env.close()


if __name__ == "__main__":
    main()
