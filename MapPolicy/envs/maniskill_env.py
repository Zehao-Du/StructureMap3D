from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Optional

# 兼容直接脚本运行：
# python MapPolicy/envs/maniskill_wrapper_env.py
# 此时 Python 不会自动把项目根目录加入 sys.path，导致 `import MapPolicy` 失败。
if __package__ is None or __package__ == "":
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

import gymnasium
import numpy as np
import torch
import tqdm
from gymnasium import Wrapper
from termcolor import colored
import mani_skill.envs
from mani_skill.utils.structs import Actor, Link

from MapPolicy.envs.evaluator import Evaluator
from MapPolicy.helpers.gymnasium import VideoWrapper
from MapPolicy.helpers.graphics import PointCloud

MANISKILL_LANGUAGE_DESCRIPTION = {
    "PickCube-v1": "Grasp a red cube with the Panda robot and move it to a target goal position.",
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PlaceSphere-v1": "Place the sphere into the shallow bin.",
    "PlugCharger-v1": "Pick up the charger and plug it into the socket.", # Official: The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.
    "PullCubeTool-v1": "Given an L-shaped tool that is within the reach of the robot, leverage the tool to pull a cube that is out of it's reach",
    "StackCube-v1": "Pick up a red cube and stack it on top of a green cube",
    "StackPyramid-v1": "Pick up a red cube, place it next to the green cube, and stack the blue cube on top of the red and green cubes"
}

class ManiSkillEnv(gymnasium.Env):
    """ManiSkill environment adapter.

    目标：把 ManiSkill 的原生 obs（可能是 nested dict / torch tensor / vectorized env）
    统一适配成 StructureMap3D 训练/采数管线使用的 obs_dict：

    - image: (n, H, W, 3) uint8
    - point_cloud: (n, num_points, 6) float32  -> xyzrgb
    - point_cloud_no_robot: (n, num_points, 6) float32  -> xyzrgb
    - robot_state: (n, D) float32
    - raw_state: (n, K) float32
    """

    def __init__(
        self,
        task_name: str,
        max_episode_length: int = 200,
        image_size: int = 224,
        camera_name: str = "base_camera",
        obs_mode: str = "rgb+depth+segmentation",
        control_mode: str = "pd_joint_pos",
        use_point_crop: bool = True,
        num_points: int = 1024,
        point_sample_method: str = "fps",
        render_mode: Optional[str] = None,
        num_envs: int = 1,
    ):
        super().__init__()

        self.task_id = task_name
        self.max_episode_length = max_episode_length
        self.image_size = image_size
        self.camera_name = camera_name
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.use_point_crop = use_point_crop
        self.num_points = num_points
        self.point_sample_method = point_sample_method
        self.render_mode = render_mode
        self.num_envs = num_envs

        self.cur_step = 0

        if self.render_mode is None:
            self.render_mode = "rgb_array"

        self.env = gymnasium.make(
            task_name,
            num_envs=num_envs,
            max_episode_steps=max_episode_length,
            obs_mode=self.obs_mode,
            control_mode=control_mode,
            render_mode=self.render_mode,
            sensor_configs=dict(width=image_size, height=image_size),
        )

        # Expose spaces to match Gymnasium API expectations
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Optional: a simple language/text field for downstream code parity
        self.text = MANISKILL_LANGUAGE_DESCRIPTION[task_name]

        raw_obs, info = self.env.reset()
        raw_obs = self._squeeze_batch_dim(raw_obs)
        self._last_raw_obs = raw_obs
        self._last_info = self._squeeze_batch_dim(info)
        

    # -----------------------------
    # Helpers (conversion utilities)
    # -----------------------------
    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        """Convert torch Tensor (cpu/cuda) or numpy-like to np.ndarray on CPU."""
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _to_scalar(x: Any):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        arr = np.asarray(x)
        if arr.size == 1:
            return arr.item()
        return arr

    @staticmethod
    def _flatten_to_1d(x: Any) -> np.ndarray:
        if isinstance(x, dict):
            chunks = [ManiSkillEnv._flatten_to_1d(v) for _, v in sorted(x.items())]
            chunks = [c for c in chunks if c.size > 0]
            if len(chunks) == 0:
                return np.zeros((0,), dtype=np.float32)
            return np.concatenate(chunks, axis=0).astype(np.float32)
        arr = ManiSkillEnv._to_numpy(x).astype(np.float32)
        return arr.reshape(-1)

    def _extract_success(self, info: Any) -> bool:
        info = self._squeeze_batch_dim(info)
        if not isinstance(info, dict):
            return False
        for key in ("success", "is_success", "episode_success", "task_success", "successes"):
            if key in info:
                try:
                    return bool(self._to_scalar(info[key]))
                except Exception:
                    return False
        return False

    def _build_obs_dict(self) -> dict[str, Any]:
        image = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud = self.get_point_cloud()
        point_cloud_no_robot = self.get_point_cloud_no_robot()

        obs_dict = {
            "image": image,
            "robot_state": robot_state,
            "point_cloud": point_cloud,
            "point_cloud_no_robot": point_cloud_no_robot,
        }
        return obs_dict
    
    def rgbd_to_xyzrgb_batch(
        self,
        remove_seg_ids: Optional[set[int]] = None,
    ) -> np.ndarray:
        """
        批量将RGBD（深度已转米）+相机参数转换为xyzrgb数组
        Args:
            rgb: RGB图像，shape=(B, H, W, 3) 或 (T, H, W, 3)，dtype=uint8
            depth_m: 深度图（米），shape=(B, H, W) 或 (T, H, W)，dtype=float32
            intrinsic_cv: 相机内参（CV格式），shape=(B, 3, 3)
            cam2world_gl: 相机外参（GL格式），shape=(B, 4, 4)
        Returns:
            xyzrgb_batch: 批量点云，shape=(B, N_max, 6)，N_max为单个batch最大有效点数，不足补0
            valid_counts: 每个batch的有效点数列表，长度=B
        """
        # fetch tensors directly; ensure torch dtype and device
        rgb = self.get_rgb()
        depth_m = self.get_depth()
        # convert to tensors (defaults to cpu) first
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.tensor(rgb)
        if not isinstance(depth_m, torch.Tensor):
            depth_m = torch.tensor(depth_m)

        # fetch camera parameters (may already live on cuda)
        sensor_param = self._last_raw_obs['sensor_param'][self.camera_name]
        intrinsic_cv = torch.as_tensor(sensor_param['intrinsic_cv'])
        cam2world_gl = torch.as_tensor(sensor_param['cam2world_gl'])

        # ensure all tensors are on the same device (prefer the device of intrinsics)
        device = intrinsic_cv.device
        if rgb.device != device:
            rgb = rgb.to(device)
        if depth_m.device != device:
            depth_m = depth_m.to(device)
        intrinsic_cv = intrinsic_cv.to(device)
        cam2world_gl = cam2world_gl.to(device)

        # log shapes
        print(f"[DEBUG] rgb tensor shape={tuple(rgb.shape)}, depth tensor shape={tuple(depth_m.shape)}")
        print(f"[DEBUG] intrinsic_cv tensor shape={tuple(intrinsic_cv.shape)}, device={device}")

        # if depth has channel dimension drop it
        if depth_m.ndim == 4 and depth_m.size(-1) == 1:
            depth_m = depth_m.squeeze(-1)
            print(f"[DEBUG] squeezed depth channel, new shape {tuple(depth_m.shape)}")

        B, H, W = rgb.shape[0], rgb.shape[1], rgb.shape[2]
        print(f"[DEBUG] using B={B}, H={H}, W={W}")

        # 1. make uv grid with torch (use float to avoid dtype mismatch)
        u = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
        v = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
        u = u.unsqueeze(0).expand(B, H, W)
        v = v.unsqueeze(0).expand(B, H, W)

        # ensure intrinsics have batch dim
        if intrinsic_cv.ndim == 2:
            intrinsic_cv = intrinsic_cv.unsqueeze(0)
        if intrinsic_cv.size(0) != B:
            intrinsic_cv = intrinsic_cv.expand(B, -1, -1)
        if cam2world_gl.ndim == 2:
            cam2world_gl = cam2world_gl.unsqueeze(0)
        if cam2world_gl.size(0) != B:
            cam2world_gl = cam2world_gl.expand(B, -1, -1)

        intrinsic_inv = torch.inverse(intrinsic_cv)
        pix_coords = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(B, -1, 3)
        cam_coords = depth_m.reshape(B, -1, 1) * (pix_coords @ intrinsic_inv.transpose(1, 2))
        
        # -------------------------- 3. 相机→世界坐标系 --------------------------
        # cam_coords shape: (B, H*W, 3); add homogeneous 1 and transform
        ones = torch.ones((B, H * W, 1), device=cam_coords.device)
        cam_coords_homo = torch.cat([cam_coords, ones], dim=-1)  # (B, H*W, 4)
        world_xyz = (cam_coords_homo @ cam2world_gl.transpose(1, 2))[:, :, :3]  # (B, H*W, 3)

        # -------------------------- 4. 拼接RGB --------------------------
        rgb_flat = rgb.reshape(B, -1, 3).float() / 255.0  # assume uint8->float
        xyzrgb = torch.cat([world_xyz, rgb_flat], dim=-1)  # (B, H*W, 6)

        # -------------------------- 5. 过滤无效点 --------------------------
        valid_masks = (depth_m.reshape(B, -1) > 0)  # bool tensor

        # optional segmentation-based exclusion
        if remove_seg_ids is not None and len(remove_seg_ids) > 0:
            seg = self._last_raw_obs['sensor_data'][self.camera_name].get('segmentation', None)
            if seg is not None:
                seg_t = torch.as_tensor(seg, device=device)
                # drop channel if present
                if seg_t.ndim == 4 and seg_t.size(-1) == 1:
                    seg_t = seg_t.squeeze(-1)
                seg_flat = seg_t.reshape(B, -1)
                remove_tensor = torch.tensor(list(remove_seg_ids), device=device, dtype=seg_flat.dtype)
                keep_seg = ~torch.isin(seg_flat, remove_tensor)
                valid_masks = valid_masks & keep_seg

        valid_counts = [mask.sum().item() for mask in valid_masks]
        N_max = max(valid_counts) if valid_counts else 0

        # allocate output numpy array and fill
        xyzrgb_batch = torch.zeros((B, N_max, 6), dtype=torch.float32, device=xyzrgb.device)
        for b in range(B):
            pts = xyzrgb[b][valid_masks[b]]  # (N_b,6)
            xyzrgb_batch[b, : pts.shape[0]] = pts
        
        return xyzrgb_batch.cpu().numpy()


    # -----------------------------
    # Core getters
    # -----------------------------
    def get_robot_state(self) -> np.ndarray:
        """Extract [tcp_pos(3), gripper_width(1)] from ManiSkill 3 observation."""
        obs = self._last_raw_obs
        # tcp_pose is in 'extra', qpos is in 'agent'
        tcp_pos = self._to_numpy(obs["extra"]["tcp_pose"])[..., :3]
        qpos = self._to_numpy(obs["agent"]["qpos"])
        # qpos is (9,) after squeeze. Panda gripper are last two dimensions.
        gripper_width = qpos[..., -1:] + qpos[..., -2:-1]
        state = np.concatenate([tcp_pos.reshape(-1), gripper_width.reshape(-1)], axis=-1)
        return state.astype(np.float32)

    def get_rgb(self) -> np.ndarray:
        """Return RGB image (n, H, W, 3) uint8 when available.

        The underlying ManiSkill env may return torch tensors; we convert to
        numpy first so callers can safely call ``astype`` etc.
        """
        sensor_data = self._last_raw_obs['sensor_data']
        rgb = sensor_data[self.camera_name]["rgb"]
        rgb_np = self._to_numpy(rgb).astype(np.uint8)
        return rgb_np
    
    def get_depth(self) -> np.ndarray:
        """Return depth image (n, H, W) float32 when available."""
        sensor_data = self._last_raw_obs['sensor_data']
        depth_int = sensor_data[self.camera_name]["depth"]
        depth_np = self._to_numpy(depth_int).astype(np.float32)
        depth_float = depth_np / 1000.0  # depth is in millimeters, convert to meters
        return depth_float
    
    def _get_segmentation_ids(self):
        """Internal helper to identify IDs for filtering or robot masking."""
        base = getattr(self.env, "unwrapped", self.env)
        seg_map = getattr(base, "segmentation_id_map", {})
        
        link_ids = set()
        ground_ids = set()
        table_workspace_ids = set()
        
        for obj_id, obj in seg_map.items():
            name = str(getattr(obj, "name", "")).lower()
            obj_id = int(obj_id)
            # Use Link for robot components (Panda/Gripper links)
            if isinstance(obj, Link) and not isinstance(obj, Actor):
                link_ids.add(obj_id)
            if isinstance(obj, Actor):
                if "ground" in name:
                    ground_ids.add(obj_id)
                elif "table-workspace" in name:
                    table_workspace_ids.add(obj_id)
        return link_ids, ground_ids, table_workspace_ids

    def get_point_cloud(self, filter_table_workspace: bool = False) -> np.ndarray:
        """返回采样后的点云 (num_points, 6)，字段为 xyzrgb。

        The original implementation did not filter any segmentation ids, which
        meant the ground plane might be included.  Here we remove points whose
        segmentation id corresponds to the ground actor.  The caller can still
        opt into removing the table workspace as well by setting
        ``filter_table_workspace`` to True (default).
        """
        # gather segmentation ids; we always drop ground
        _, ground_ids, table_ids = self._get_segmentation_ids()
        remove_ids: set[int] = set(ground_ids)
        if filter_table_workspace:
            remove_ids |= set(table_ids)
        return self.rgbd_to_xyzrgb_batch(remove_seg_ids=remove_ids)

    def get_point_cloud_no_robot(self, filter_table_workspace: bool = False) -> np.ndarray:
        """Return point cloud after removing robot (and optionally table) via segmentation.

        This implementation uses the rgbd conversion path with an explicit
        segmentation image rather than the legacy full point cloud provided by
        ManiSkill. The caller can disable table workspace removal.
        """
        link_ids, ground_ids, table_ids = self._get_segmentation_ids()
        remove_ids = set(link_ids) | set(ground_ids)
        if filter_table_workspace:
            remove_ids |= set(table_ids)
        # perform conversion with filter
        return self.rgbd_to_xyzrgb_batch(remove_seg_ids=remove_ids)

    def get_obs(self) -> dict[str, Any]:
        """对外统一接口，字段与 MetaWorldEnv 保持一致。"""
        return self._build_obs_dict()

    # -----------------------------
    # Gymnasium API
    # -----------------------------
    def _squeeze_batch_dim(self, x: Any) -> Any:
        """Squeeze leading batch dim when num_envs==1 for nested dict/arrays/tensors."""
        if self.num_envs != 1:
            return x
        if isinstance(x, dict):
            return {k: self._squeeze_batch_dim(v) for k, v in x.items()}
        if hasattr(x, "shape") and len(x.shape) >= 1 and x.shape[0] == 1:
            if hasattr(x, "detach"):
                return x.squeeze(0)
            else:
                return np.squeeze(x, axis=0)
        return x

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Gym reset -> returns obs_dict.

        Ensure that if underlying env returns batched obs (num_envs=1), we squeeze it to match MetaWorldEnv.
        """
        super().reset(seed=seed, options=options)
        self.cur_step = 0
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._squeeze_batch_dim(obs)
        self._last_raw_obs = obs
        self._last_info = self._squeeze_batch_dim(info)
        return self.get_obs()

    def step(self, action: np.ndarray):
        """Gym step -> returns (obs_dict, reward, terminated, truncated, info)."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._squeeze_batch_dim(obs)
        reward = self._to_scalar(self._squeeze_batch_dim(reward))
        terminated = bool(self._to_scalar(self._squeeze_batch_dim(terminated)))
        truncated = bool(self._to_scalar(self._squeeze_batch_dim(truncated)))
        info = self._squeeze_batch_dim(info)

        self._last_raw_obs = obs
        self._last_info = info
        self.cur_step += 1
        truncated = truncated or self.cur_step >= self.max_episode_length

        obs_dict = self.get_obs()
        if isinstance(info, dict):
            info["gripper_proprio"] = obs_dict["raw_state"][:4]
        return obs_dict, reward, terminated, truncated, info

    def close(self):
        if self.env is not None:
            self.env.close()


class ManiSkillEvaluator(Evaluator):
    """ManiSkill evaluator.

    接口和 MetaWorldEvaluator 对齐：
    - evaluate 返回 (avg_success, avg_rewards)
    - verbose=True 时保留每条轨迹视频用于 wandb 可视化

    现在支持 ``num_envs`` 参数，用于并行评估多个环境以加速。
    评估循环会在内部维护一组单环境实例，并且批量执行模型前向。
    """

    def __init__(
        self,
        task_id: Optional[str] = None,
        task_name: Optional[str] = None,
        max_episode_length: int = 200,
        image_size: int = 128,
        camera_name: str = "base_camera",
        obs_mode: str = "pointcloud",
        control_mode: str = "pd_joint_pos",
        use_point_crop: bool = False, # No use at current time
        num_points: int = 1024,
        point_cloud_camera_names: Optional[list[str]] = None,# No use at current time
        point_sample_method: str = "fps",
        render_mode: Optional[str] = None,
        num_envs: int = 1,
    ):
        if task_id is None:
            task_id = task_name
        if task_id is None:
            raise ValueError("ManiSkillEvaluator requires either task_id or task_name.")

        self.num_envs = num_envs
        # create a list of independent single-env instances. this is easier than
        # trying to manage gym.vector semantics in evaluation.
        self.envs = [
            ManiSkillEnv(
                task_id=task_id,
                max_episode_length=max_episode_length,
                image_size=image_size,
                camera_name=camera_name,
                obs_mode=obs_mode,
                control_mode=control_mode,
                num_points=num_points,
                point_sample_method=point_sample_method,
                render_mode=render_mode,
                num_envs=1,
            )
            for _ in range(num_envs)
        ]
        # video wrapper on each
        self.envs = [VideoWrapper(e) for e in self.envs]
        # expose one env for compatibility (used elsewhere?)
        self.env = self.envs[0]

    def evaluate(self, num_episodes, policy, verbose: bool = False):
        task_id = Wrapper.get_wrapper_attr(self.env, "task_id")
        # policy is assumed to take batched inputs; we'll aggregate observations

        # bookkeeping for results
        success_list = []
        rewards_list = []
        video_steps_list = []

        # helper batch runner that steps a list of envs until each finishes once
        def _run_batch(envs, policy, verbose):
            batch_size = len(envs)
            obs_list = [e.reset() for e in envs]
            dones = [False] * batch_size
            rewards = [0.0] * batch_size
            successes = [False] * batch_size
            frames = None

            while not all(dones):
                # build batch input for active envs
                imgs = []
                pcs = []
                pc_nrs = []
                rstates = []
                texts = []
                idx_map = []  # mapping from batch index to env index
                for i, o in enumerate(obs_list):
                    if dones[i]:
                        continue
                    imgs.append(o["image"])
                    pcs.append(o["point_cloud"])
                    # pc_nrs.append(o["point_cloud_no_robot"])  # temporarily disabled
                    rstates.append(o["robot_state"])
                    texts.append(envs[i].text)
                    idx_map.append(i)
                if len(imgs) == 0:
                    break
                device = next(policy.parameters()).device
                imgs_t = torch.from_numpy(np.stack(imgs)).float().to(device).permute(0, 3, 1, 2)
                pcs_t = torch.from_numpy(np.stack(pcs)).float().to(device)
                # pc_nrs_t = torch.from_numpy(np.stack(pc_nrs)).float().to(device)
                rstates_t = torch.from_numpy(np.stack(rstates)).float().to(device)

                input_data = {
                    "images": imgs_t,
                    "point_clouds": pcs_t,
                    # "point_cloud_no_robot": pc_nrs_t,  # disabled for now
                    "robot_states": rstates_t,
                    "texts": texts,
                }
                with torch.no_grad():
                    actions_batch = policy(**input_data)
                if isinstance(actions_batch, torch.Tensor) and torch.isnan(actions_batch).any():
                    warnings.warn("NaNs in batch policy output; zeroing", RuntimeWarning)
                    actions_batch = torch.nan_to_num(actions_batch, nan=0.0)
                actions_np = actions_batch.to("cpu").detach().numpy()

                # step each env
                for bi, env_idx in enumerate(idx_map):
                    action = actions_np[bi]
                    if np.isnan(action).any():
                        action = np.nan_to_num(action, nan=0.0)

                    o, r, t, tr, info = envs[env_idx].step(action)
                    obs_list[env_idx] = o
                    rewards[env_idx] += float(r)
                    successes[env_idx] = (
                        successes[env_idx]
                        or Wrapper.get_wrapper_attr(envs[env_idx], "_extract_success")(
                            info
                        )
                    )
                    if t or tr:
                        dones[env_idx] = True

            if verbose:
                frames = [e.get_frames().transpose(0, 3, 1, 2) for e in envs]
            return successes, rewards, frames

        idx = 0
        pbar = tqdm.tqdm(total=num_episodes, desc=f"Evaluating ManiSkill <{colored(task_id, 'red')}>")
        while idx < num_episodes:
            batch = min(self.num_envs, num_episodes - idx)
            succs, rews, frs = _run_batch(self.envs[:batch], policy, verbose)
            success_list.extend(succs)
            rewards_list.extend(rews)
            if verbose:
                for env_frames in frs:
                    video_steps_list.append(env_frames)
            idx += batch
            pbar.update(batch)
        pbar.close()

        avg_success = sum(success_list) / num_episodes
        avg_rewards = sum(rewards_list) / num_episodes
        if verbose:
            self.success_list = success_list
            self.rewards_list = rewards_list
            self.video_steps_list = video_steps_list
        return avg_success, avg_rewards

    def callback_verbose(self, wandb_logger):
        import plotly.express as px
        import plotly.graph_objects as go
        import wandb

        fig1 = go.Figure(
            data=[
                go.Bar(
                    x=["Success", "Failure"],
                    y=[
                        sum(self.success_list),
                        len(self.success_list) - sum(self.success_list),
                    ],
                )
            ]
        )
        fig2 = px.box(self.rewards_list, title="Rewards distribution")
        wandb_logger.log({"Charts/success_failure": fig1})
        wandb_logger.log({"Charts/rewards_distribution": fig2})

        for i, (success, rewards, video_steps) in enumerate(
            zip(self.success_list, self.rewards_list, self.video_steps_list)
        ):
            if success:
                wandb_logger.log(
                    {
                        f"validation/video_steps_success_{i}": wandb.Video(
                            video_steps, fps=30
                        ),
                    }
                )
            else:
                wandb_logger.log(
                    {
                        f"validation/video_steps_failure_{i}": wandb.Video(
                            video_steps, fps=30
                        ),
                    }
                )


if __name__ == "__main__":
    # simplified entrypoint that always saves a pair of point cloud files
    import argparse
    import faulthandler
    import os
    import pathlib
    import sys

    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("MAP_POLICY_FPS_DEVICE", "cpu")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    np.random.seed(0)
    torch.manual_seed(0)
    faulthandler.enable()

    parser = argparse.ArgumentParser(description="Debug ManiSkillEnv and export point clouds")
    parser.add_argument("--task_name", type=str, default="PlugCharger-v1")
    parser.add_argument("--camera_name", type=str, default="hand_camera")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_points", type=int, default=16384)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data2/lirui/maniskill_debug",
        help="directory where rgb image and point cloud PLYs are written",
    )
    args = parser.parse_args()

    def log(msg: str):
        print(msg, flush=True)

    env = None
    try:
        log("[INFO] creating environment")
        env = ManiSkillEnv(
            task_name=args.task_name,
            max_episode_length=200,
            image_size=args.image_size,
            camera_name=args.camera_name,
            control_mode="pd_joint_pos",
            num_points=args.num_points,
            point_sample_method="fps",
            render_mode="rgb_array",
            num_envs=2,
        )

        log("[INFO] resetting env")
        obs = env.reset(seed=0)

        log("[INFO] extracting point clouds")
        pc_with_robot = env.get_point_cloud()
        pc_no_robot = env.get_point_cloud_no_robot()
        rgb = env.get_rgb()
        # report number of valid points in each batch element
        if isinstance(pc_with_robot, np.ndarray) and pc_with_robot.ndim == 3:
            counts = [np.count_nonzero(np.linalg.norm(pc_with_robot[b,:,:3], axis=1)) for b in range(pc_with_robot.shape[0])]
            log(f"[INFO] point counts (with robot) per env: {counts}")
        else:
            log(f"[INFO] point count (with robot): {pc_with_robot.shape[0] if hasattr(pc_with_robot, 'shape') else 'unknown'}")
        if isinstance(pc_no_robot, np.ndarray) and pc_no_robot.ndim == 3:
            counts_nr = [np.count_nonzero(np.linalg.norm(pc_no_robot[b,:,:3], axis=1)) for b in range(pc_no_robot.shape[0])]
            log(f"[INFO] point counts (no robot) per env: {counts_nr}")
        else:
            log(f"[INFO] point count (no robot): {pc_no_robot.shape[0] if hasattr(pc_no_robot, 'shape') else 'unknown'}")

        save_dir = pathlib.Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # save rgb for reference (handle batch dimension)
        rgb_path = save_dir / "rgb.png"
        try:
            import imageio.v2 as imageio
            if isinstance(rgb, np.ndarray) and rgb.ndim == 4:
                # multiple envs: save each with index suffix
                for b in range(rgb.shape[0]):
                    out = save_dir / f"rgb_{b}.png"
                    imageio.imwrite(str(out), rgb[b])
                    log(f"[INFO] saved rgb to {out}")
            else:
                imageio.imwrite(str(rgb_path), rgb)
                log(f"[INFO] saved rgb to {rgb_path}")
        except Exception as exc:
            log(f"[WARN] failed to save rgb: {exc}")

        def _save_point_cloud_ply(np_pc: np.ndarray, out_path: pathlib.Path):
            """Write a (N,6) xyzrgb array to an ASCII PLY file.

            The input ``np_pc`` may have rgb values in [0,1] or [0,255].
            We coerce to uint8 scalars and iterate safely to avoid numpy-array
            scalar conversion issues.
            """
            xyz = np_pc[:, :3].astype(np.float32)
            rgb_local = np_pc[:, 3:6].astype(np.float32)
            if rgb_local.size > 0 and rgb_local.max() <= 1.01:
                rgb_local = rgb_local * 255.0
            rgb_local = np.clip(rgb_local, 0.0, 255.0).astype(np.uint8)

            with open(out_path, "w", encoding="utf-8") as f:
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
                    # use Python scalars to avoid nested-array error
                    ri = int(rgb_local[i, 0])
                    gi = int(rgb_local[i, 1])
                    bi = int(rgb_local[i, 2])
                    f.write(f"{x} {y} {z} {ri} {gi} {bi}\n")

        # helper to write batched point clouds
        def _maybe_save_batch(np_pc: np.ndarray, base_path: pathlib.Path):
            if np_pc.ndim == 3:
                for b in range(np_pc.shape[0]):
                    out = base_path.with_name(f"{base_path.stem}_{b}{base_path.suffix}")
                    _save_point_cloud_ply(np_pc[b], out)
                    log(f"[INFO] saved point cloud to {out}")
            else:
                _save_point_cloud_ply(np_pc, base_path)
                log(f"[INFO] saved point cloud to {base_path}")


        with_robot_ply = save_dir / "pc_with_robot.ply"
        no_robot_ply = save_dir / "pc_no_robot.ply"
        _maybe_save_batch(pc_with_robot, with_robot_ply)
        _maybe_save_batch(pc_no_robot, no_robot_ply)

    finally:
        if env is not None:
            env.close()
            log("[INFO] env closed")
