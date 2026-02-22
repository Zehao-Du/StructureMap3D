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

class ManiSkillEnv(gymnasium.Env):
    """ManiSkill environment adapter.

    目标：把 ManiSkill 的原生 obs（可能是 nested dict / torch tensor / vectorized env）
    统一适配成 StructureMap3D 训练/采数管线使用的 obs_dict：

    - image: (H, W, 3) uint8
    - point_cloud: (num_points, 6) float32  -> xyzrgb
    - point_cloud_no_robot: (num_points, 6) float32  -> xyzrgb
    - robot_state: (D,) float32
    - raw_state: (K,) float32
    """

    def __init__(
        self,
        task_id: str,
        max_episode_length: int = 200,
        image_size: int = 224,
        camera_name: str = "base_camera",
        obs_mode: str = "pointcloud",
        control_mode: str = "pd_joint_pos",
        use_point_crop: bool = False,
        num_points: int = 1024,
        point_sample_method: str = "fps",
        render_mode: Optional[str] = None,
        num_envs: int = 1,
    ):
        super().__init__()

        self.task_id = task_id
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
            task_id,
            num_envs=num_envs,
            max_episode_steps=max_episode_length,
            obs_mode=self.obs_mode,
            control_mode=control_mode,
            render_mode=self.render_mode,
            sensor_configs={"width": image_size, "height": image_size},
        )

        # Expose spaces to match Gymnasium API expectations
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Optional: a simple language/text field for downstream code parity
        self.text = task_id

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
        raw_state = self.get_raw_state()
        point_cloud = self.get_point_cloud()
        point_cloud_no_robot = self.get_point_cloud_no_robot()

        obs_dict = {
            "image": image,
            "robot_state": robot_state,
            "raw_state": raw_state,
            "point_cloud": point_cloud,
            "point_cloud_no_robot": point_cloud_no_robot,
        }
        return obs_dict

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

    def get_raw_state(self) -> np.ndarray:
        """Return concatenated agent and extra observations vector."""
        obs = self._last_raw_obs
        agent_flat = self._flatten_to_1d(obs["agent"])
        extra_flat = self._flatten_to_1d(obs["extra"])
        return np.concatenate([agent_flat, extra_flat]).astype(np.float32)

    def get_rgb(self) -> np.ndarray:
        """Return RGB image (H, W, 3) uint8 when available.

        Minimal implementation: try common keys; if not present raise NotImplementedError.
        """
        rendered = self.env.render()
        if rendered is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        rgb = self._to_numpy(rendered)

        # ManiSkill 在 num_envs=1 时，部分路径会返回 (1, H, W, C)。
        # 这里统一压缩为 (H, W, C)，便于后续保存/可视化。
        rgb = self._squeeze_batch_dim(rgb)

        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        if rgb.ndim == 3 and rgb.shape[-1] > 3:
            rgb = rgb[..., :3]

        if rgb.dtype != np.uint8:
            rgb = np.asarray(rgb, dtype=np.float32)
            if rgb.max() <= 1.01:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb
        
    def _build_full_point_cloud_and_mask(self):
        """Standard ManiSkill 3 pointcloud extractor (xyzrgb)."""
        pc = self._last_raw_obs["pointcloud"]
        xyzw = self._to_numpy(pc["xyzw"]).astype(np.float32)
        rgb = self._to_numpy(pc["rgb"]).astype(np.float32)

        xyz = xyzw[..., :3]
        # w > 0 means valid point in SAPIEN
        valid_mask = xyzw[..., 3] > 0
        
        if rgb.max() <= 1.01:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.float32)

        point_cloud = np.concatenate([xyz, rgb], axis=-1)
        return point_cloud, valid_mask, pc

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

    def get_point_cloud(self, filter_table_workspace: bool = True) -> np.ndarray:
        """返回采样后的点云 (num_points, 6)，字段为 xyzrgb。"""
        point_cloud, valid_mask, pc_src = self._build_full_point_cloud_and_mask()
        seg = self._to_numpy(pc_src["segmentation"]).reshape(-1)

        _, ground_ids, table_ids = self._get_segmentation_ids()
        remove_ids = ground_ids
        if filter_table_workspace:
            remove_ids |= table_ids

        keep_mask = (~np.isin(seg, np.array(list(remove_ids), dtype=seg.dtype))) & valid_mask
        point_cloud = PointCloud.point_cloud_sampling(
            point_cloud[keep_mask], self.num_points, self.point_sample_method
        )
        return point_cloud.astype(np.float32)

    def get_point_cloud_no_robot(self, filter_table_workspace: bool = True) -> np.ndarray:
        """基于 ManiSkill 官方 Actor/Link 语义过滤机器人点云。"""
        full, valid_mask, pc_src = self._build_full_point_cloud_and_mask()
        seg = self._to_numpy(pc_src["segmentation"]).reshape(-1)

        link_ids, ground_ids, table_ids = self._get_segmentation_ids()
        remove_ids = link_ids | ground_ids
        if filter_table_workspace:
            remove_ids |= table_ids

        keep_mask = (~np.isin(seg, np.array(list(remove_ids), dtype=seg.dtype))) & valid_mask
        point_cloud = PointCloud.point_cloud_sampling(
            full[keep_mask], self.num_points, self.point_sample_method
        )
        return point_cloud.astype(np.float32)

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
                    pc_nrs.append(o["point_cloud_no_robot"])
                    rstates.append(o["robot_state"])
                    texts.append(envs[i].text)
                    idx_map.append(i)
                if len(imgs) == 0:
                    break
                device = next(policy.parameters()).device
                imgs_t = torch.from_numpy(np.stack(imgs)).float().to(device).permute(0, 3, 1, 2)
                pcs_t = torch.from_numpy(np.stack(pcs)).float().to(device)
                pc_nrs_t = torch.from_numpy(np.stack(pc_nrs)).float().to(device)
                rstates_t = torch.from_numpy(np.stack(rstates)).float().to(device)

                input_data = {
                    "images": imgs_t,
                    "point_clouds": pcs_t,
                    "point_cloud_no_robot": pc_nrs_t,
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

    parser = argparse.ArgumentParser(description="Staged debug for ManiSkill wrapper")
    parser.add_argument("--task_id", type=str, default="PlugCharger-v1")
    parser.add_argument("--camera_name", type=str, default="base_camera")
    parser.add_argument("--image_size", type=int, default=480)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--obs_mode", type=str, default="pointcloud")
    parser.add_argument("--save_dir", type=str, default="data2/lirui/maniskill_debug")
    parser.add_argument(
        "--stage",
        type=int,
        default=4,
        help="1: create env; 2: + reset; 3: + read raw pointcloud; 4: + render rgb and wrapper pointcloud; 5: + call getters and print datatypes; 6: + one step and save second rgb",
    )
    args = parser.parse_args()

    def log_stage(msg: str):
        print(msg, flush=True)

    env = None
    obs_dict = None

    try:
        log_stage("[STAGE-0] Start debug script")
        log_stage(f"[CFG] task_id={args.task_id}, camera_name={args.camera_name}, image_size={args.image_size}, num_points={args.num_points}, stage={args.stage}")

        log_stage("[STAGE-1] Creating ManiSkillEnv...")
        env = ManiSkillEnv(
            task_id=args.task_id,
            max_episode_length=200,
            image_size=args.image_size,
            camera_name=args.camera_name,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            num_points=args.num_points,
            point_sample_method="fps",
            render_mode="rgb_array",
            num_envs=1,
        )
        log_stage("[STAGE-1] OK")

        if args.stage < 2:
            sys.exit(0)

        log_stage("[STAGE-2] Calling env.reset(seed=0)...")
        obs_dict = env.reset(seed=0)
        log_stage("[STAGE-2] OK")
        log_stage(f"[STAGE-2] obs keys: {list(obs_dict.keys())}")

        if args.stage < 3:
            sys.exit(0)

        log_stage("[STAGE-3] Reading raw pointcloud dict from _last_raw_obs...")
        raw_obs = env._last_raw_obs
        if not isinstance(raw_obs, dict) or "pointcloud" not in raw_obs:
            raise RuntimeError("raw_obs does not contain pointcloud dict")
        raw_pc = raw_obs["pointcloud"]
        raw_xyzw = env._to_numpy(raw_pc["xyzw"])
        raw_rgb = env._to_numpy(raw_pc["rgb"])
        log_stage(f"[STAGE-3] xyzw shape={raw_xyzw.shape}, dtype={raw_xyzw.dtype}")
        log_stage(f"[STAGE-3] rgb shape={raw_rgb.shape}, dtype={raw_rgb.dtype}")
        if "robot_seg" in raw_pc:
            raw_robot_seg = env._to_numpy(raw_pc["robot_seg"])
            log_stage(f"[STAGE-3] robot_seg shape={raw_robot_seg.shape}, dtype={raw_robot_seg.dtype}")
        log_stage("[STAGE-3] OK")

        if args.stage < 4:
            sys.exit(0)

        log_stage("[STAGE-4] Rendering rgb (headless) + extracting wrapper point clouds...")
        rgb = env.get_rgb()
        pc_with_robot = env.get_point_cloud()
        pc_no_robot = env.get_point_cloud_no_robot()

        save_dir = pathlib.Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存一张 RGB，便于对齐当前视角。
        rgb_path = save_dir / "debug_rgb.png"
        try:
            import imageio.v2 as imageio

            imageio.imwrite(str(rgb_path), rgb)
            log_stage(f"[STAGE-4] saved rgb png: {rgb_path}")
        except Exception as exc:
            log_stage(f"[STAGE-4] skip rgb save due to error: {exc}")

        def _save_point_cloud_ply(np_pc: np.ndarray, out_path: pathlib.Path):
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
                    r, g, b = rgb_local[i]
                    f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

        with_robot_ply = save_dir / "debug_pc_WITH_robot.ply"
        no_robot_ply = save_dir / "debug_pc_NO_robot.ply"
        _save_point_cloud_ply(pc_with_robot, with_robot_ply)
        _save_point_cloud_ply(pc_no_robot, no_robot_ply)
        log_stage(f"[STAGE-4] saved point cloud ply: {with_robot_ply}")
        log_stage(f"[STAGE-4] saved point cloud ply: {no_robot_ply}")

        log_stage(f"[STAGE-4] rgb shape={rgb.shape}, dtype={rgb.dtype}")
        log_stage(f"[STAGE-4] point_cloud shape={pc_with_robot.shape}, dtype={pc_with_robot.dtype}")
        log_stage(f"[STAGE-4] point_cloud_no_robot shape={pc_no_robot.shape}, dtype={pc_no_robot.dtype}")
        log_stage("[STAGE-4] OK")

        if args.stage < 5:
            sys.exit(0)

        log_stage("[STAGE-5] Calling getter methods and printing datatype/shape/min/max info...")

        def _value_summary(x):
            if x is None:
                return "type=NoneType"

            # 统一转 numpy，便于输出 shape/min/max。
            if hasattr(x, "detach"):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)

            dtype_str = str(arr.dtype) if hasattr(arr, "dtype") else type(x).__name__
            shape_str = str(arr.shape) if hasattr(arr, "shape") else "()"

            if arr.size == 0:
                return f"dtype={dtype_str}, shape={shape_str}, min=NA, max=NA"

            if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
                min_v = np.min(arr)
                max_v = np.max(arr)
                return f"dtype={dtype_str}, shape={shape_str}, min={min_v}, max={max_v}"

            return f"dtype={dtype_str}, shape={shape_str}, min=NA, max=NA"

        def _print_datatype_tree(prefix: str, value):
            if isinstance(value, dict):
                log_stage(f"[STAGE-5] {prefix}: dict")
                for k, v in value.items():
                    _print_datatype_tree(f"{prefix}.{k}", v)
            else:
                log_stage(f"[STAGE-5] {prefix}: {_value_summary(value)}")

        getter_calls = [
            ("get_rgb", env.get_rgb),
            ("get_robot_state", env.get_robot_state),
            ("get_point_cloud", env.get_point_cloud),
            ("get_point_cloud_no_robot", env.get_point_cloud_no_robot),
            ("get_obs", env.get_obs),
        ]

        for getter_name, getter_fn in getter_calls:
            value = getter_fn()
            if isinstance(value, dict):
                _print_datatype_tree(getter_name, value)
            else:
                log_stage(f"[STAGE-5] {getter_name}: {_value_summary(value)}")

        log_stage("[STAGE-5] OK")

        if args.stage < 6:
            sys.exit(0)

        log_stage("[STAGE-6] Calling env.step(action_space.sample()) and checking outputs...")
        action = env.action_space.sample()
        log_stage(f"[STAGE-6] action shape={np.asarray(action).shape}, dtype={np.asarray(action).dtype}")

        obs_next, reward, terminated, truncated, info = env.step(action)
        log_stage(f"[STAGE-6] reward={reward}, terminated={terminated}, truncated={truncated}")
        if isinstance(info, dict):
            log_stage(f"[STAGE-6] info keys: {list(info.keys())}")

        rgb2 = obs_next["image"]
        rgb2_path = save_dir / "debug_rgb_step1.png"
        try:
            import imageio.v2 as imageio

            imageio.imwrite(str(rgb2_path), rgb2)
            log_stage(f"[STAGE-6] saved second rgb png: {rgb2_path}")
        except Exception as exc:
            log_stage(f"[STAGE-6] skip second rgb save due to error: {exc}")

        log_stage(
            f"[STAGE-6] next_obs image shape={obs_next['image'].shape}, dtype={obs_next['image'].dtype}"
        )
        log_stage(
            f"[STAGE-6] next_obs point_cloud shape={obs_next['point_cloud'].shape}, dtype={obs_next['point_cloud'].dtype}"
        )
        log_stage(
            f"[STAGE-6] next_obs point_cloud_no_robot shape={obs_next['point_cloud_no_robot'].shape}, dtype={obs_next['point_cloud_no_robot'].dtype}"
        )
        log_stage("[STAGE-6] OK")

    finally:
        if env is not None:
            env.close()
            log_stage("[CLEANUP] env closed")
