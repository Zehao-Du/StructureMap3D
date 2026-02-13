import argparse
import copy
import os
import pathlib
import sys
from typing import Any

import numpy as np
import tqdm
import zarr
from numcodecs import MsgPack
from termcolor import colored, cprint
from mani_skill.examples.motionplanning.panda.solutions import (
    solveDrawSVG,
    solveDrawTriangle,
    solveLiftPegUpright,
    solvePegInsertionSide,
    solvePickCube,
    solvePlaceSphere,
    solvePlugCharger,
    solvePullCube,
    solvePullCubeTool,
    solvePushCube,
    solveStackCube,
    solveStackPyramid,
)

# 兼容直接脚本运行：
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from MapPolicy.dataset.metaworld_dataset import MetaWorldDataset
from MapPolicy.envs.maniskill_wrapper_env import ManiSkillEnv
from MapPolicy.helpers.Common import (
    save_point_cloud_ply,
    save_rgb_image,
    save_video_imageio,
)
from MapPolicy.helpers.Logger import Logger


MP_SOLUTIONS = {
    "DrawTriangle-v1": solveDrawTriangle,
    "PickCube-v1": solvePickCube,
    "StackCube-v1": solveStackCube,
    "PegInsertionSide-v1": solvePegInsertionSide,
    "PlugCharger-v1": solvePlugCharger,
    "PlaceSphere-v1": solvePlaceSphere,
    "PushCube-v1": solvePushCube,
    "PullCubeTool-v1": solvePullCubeTool,
    "LiftPegUpright-v1": solveLiftPegUpright,
    "PullCube-v1": solvePullCube,
    "DrawSVG-v1": solveDrawSVG,
    "StackPyramid-v1": solveStackPyramid,
}


def _maybe_squeeze(x: Any):
    if isinstance(x, dict):
        return {k: _maybe_squeeze(v) for k, v in x.items()}
    if hasattr(x, "shape") and len(getattr(x, "shape")) >= 1 and getattr(x, "shape")[0] == 1:
        if hasattr(x, "detach"):
            return x.squeeze(0)
        try:
            return np.squeeze(x, axis=0)
        except Exception:
            return x
    return x


def _extract_success(info: Any) -> bool:
    info = _maybe_squeeze(info)
    if not isinstance(info, dict):
        return False
    for key in ("success", "is_success", "episode_success", "task_success", "successes"):
        if key in info:
            val = info[key]
            try:
                if hasattr(val, "detach"):
                    val = val.detach().cpu().numpy()
                val = np.asarray(val).item() if np.asarray(val).size == 1 else val
            except Exception:
                pass
            try:
                return bool(val)
            except Exception:
                return False
    return False


class _MotionPlanningCollectorEnv:
    """Proxy env: runs ManiSkill motion planner while recording per-step obs/action."""

    def __init__(self, wrapped_env: ManiSkillEnv):
        self._wrapped = wrapped_env
        self._base = wrapped_env.env
        self.records = []

    def reset(self, *args, **kwargs):
        self._wrapped.cur_step = 0
        obs, info = self._base.reset(*args, **kwargs)
        self._wrapped._last_raw_obs = self._wrapped._squeeze_batch_dim(obs)
        self._wrapped._last_info = self._wrapped._squeeze_batch_dim(info)
        self.records.clear()
        return obs, info

    def step(self, action):
        obs_before = copy.deepcopy(self._wrapped.get_obs())

        obs, reward, terminated, truncated, info = self._base.step(action)

        obs_sq = self._wrapped._squeeze_batch_dim(obs)
        reward_sq = self._wrapped._to_scalar(self._wrapped._squeeze_batch_dim(reward))
        terminated_sq = bool(self._wrapped._to_scalar(self._wrapped._squeeze_batch_dim(terminated)))
        truncated_sq = bool(self._wrapped._to_scalar(self._wrapped._squeeze_batch_dim(truncated)))
        info_sq = self._wrapped._squeeze_batch_dim(info)

        self._wrapped._last_raw_obs = obs_sq
        self._wrapped._last_info = info_sq
        self._wrapped.cur_step += 1

        truncated_sq = truncated_sq or self._wrapped.cur_step >= self._wrapped.max_episode_length

        self.records.append(
            {
                "obs": obs_before,
                "action": np.asarray(action, dtype=np.float32),
                "reward": float(reward_sq),
                "terminated": terminated_sq,
                "truncated": truncated_sq,
                "info": copy.deepcopy(info_sq),
            }
        )

        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self._base, name)


def _resolve_motionplan_solver(task_id: str):
    if task_id not in MP_SOLUTIONS:
        raise RuntimeError(
            f"No Panda motion-planning solver for task_id={task_id}. "
            f"Available: {list(MP_SOLUTIONS.keys())}"
        )
    return MP_SOLUTIONS[task_id]


def main(args):
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with arguments:'
    )
    Logger.log_info(f"Task id: {args.task_id}")
    Logger.log_info(f"Camera name: {args.camera_name}")
    Logger.log_info(f"Image size: {args.image_size}")
    Logger.log_info(f"Obs mode: {args.obs_mode}")
    Logger.log_info(f"Control mode: {args.control_mode}")
    Logger.log_info(f"Num points: {args.num_points}")
    Logger.log_info(f"Number of episodes: {args.num_episodes}")
    Logger.log_info(f"Episode length: {args.episode_length}")
    Logger.log_info(f"Save directory: {args.save_dir}")
    Logger.print_seperator()

    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    video_dir = (
        pathlib.Path(args.save_dir)
        / "visualized_data"
        / "videos"
        / args.task_id
        / args.camera_name
    )
    image_dir = (
        pathlib.Path(args.save_dir)
        / "visualized_data"
        / "images"
        / args.task_id
        / args.camera_name
    )
    point_cloud_dir = (
        pathlib.Path(args.save_dir)
        / "visualized_data"
        / "point_clouds"
        / args.task_id
        / args.camera_name
    )
    point_cloud_no_robot_dir = (
        pathlib.Path(args.save_dir)
        / "visualized_data"
        / "point_clouds_no_robot"
        / args.task_id
        / args.camera_name
    )

    video_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    point_cloud_dir.mkdir(parents=True, exist_ok=True)
    point_cloud_no_robot_dir.mkdir(parents=True, exist_ok=True)

    env = ManiSkillEnv(
        task_id=args.task_id,
        max_episode_length=args.episode_length,
        image_size=args.image_size,
        camera_name=args.camera_name,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        num_points=args.num_points,
        render_mode=None,
        num_envs=1,
    )
    solve = _resolve_motionplan_solver(args.task_id)

    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    point_cloud_no_robot_arrays = []
    robot_state_arrays = []
    raw_state_arrays = []
    action_arrays = []
    reward_arrays = []
    episode_ends_arrays = []
    env_info_arrays = []
    texts = []

    max_retries_per_scene = 3
    scene_retry_count = 0
    scene_idx = 0

    description = args.text if args.text is not None else args.task_id

    episode_idx = 0
    if args.quiet:
        process_bar = tqdm.tqdm(range(args.num_episodes))

    while episode_idx < args.num_episodes:
        # deterministic per episode if seed provided
        seed = None if args.seed is None else int(args.seed + scene_idx)

        ep_reward = 0.0
        ep_success = False
        ep_success_times = 0

        img_arrays_sub = []
        point_cloud_arrays_sub = []
        point_cloud_no_robot_arrays_sub = []
        robot_state_arrays_sub = []
        raw_state_arrays_sub = []
        action_arrays_sub = []
        reward_arrays_sub = []
        env_info_arrays_sub = []
        texts_sub = []
        total_count_sub = 0

        planner_env = _MotionPlanningCollectorEnv(env)
        try:
            res = solve(planner_env, seed=seed, debug=False, vis=False)
        except Exception as exc:
            cprint(
                f"Task: {args.task_id} Episode: {episode_idx} Scene: {scene_idx} motion-planning exception: {exc}",
                "red",
            )
            scene_retry_count += 1
            if scene_retry_count >= max_retries_per_scene:
                cprint(
                    f"Task: {args.task_id} Episode: {episode_idx} Scene: {scene_idx} reached {max_retries_per_scene} failed attempts, switching scene.",
                    "yellow",
                )
                scene_idx += 1
                scene_retry_count = 0
            continue

        if res == -1:
            cprint(
                f"Task: {args.task_id} Episode: {episode_idx} Scene: {scene_idx} failed: planner returned -1",
                "red",
            )
            scene_retry_count += 1
            if scene_retry_count >= max_retries_per_scene:
                cprint(
                    f"Task: {args.task_id} Episode: {episode_idx} Scene: {scene_idx} reached {max_retries_per_scene} failed attempts, switching scene.",
                    "yellow",
                )
                scene_idx += 1
                scene_retry_count = 0
            continue

        for rec in planner_env.records:
            obs_dict = rec["obs"]
            obs_img = obs_dict["image"]
            obs_robot_state = obs_dict["robot_state"]
            obs_raw_state = obs_dict["raw_state"]
            obs_point_cloud = obs_dict["point_cloud"]
            obs_point_cloud_no_robot = obs_dict["point_cloud_no_robot"]

            action = rec["action"]
            reward = rec["reward"]
            env_info = rec["info"]

            img_arrays_sub.append(obs_img)
            point_cloud_arrays_sub.append(obs_point_cloud)
            point_cloud_no_robot_arrays_sub.append(obs_point_cloud_no_robot)
            robot_state_arrays_sub.append(obs_robot_state)
            raw_state_arrays_sub.append(obs_raw_state)
            action_arrays_sub.append(action)
            reward_arrays_sub.append(reward)
            env_info_arrays_sub.append(env_info)
            texts_sub.append(description)

            ep_reward += float(reward)
            step_success = _extract_success(env_info)
            ep_success = ep_success or step_success
            ep_success_times += int(step_success)

        total_count_sub = len(planner_env.records)

        if total_count_sub == 0:
            cprint(
                f"Task: {args.task_id} Episode: {episode_idx} Scene: {scene_idx} failed: no trajectory recorded",
                "red",
            )
            scene_retry_count += 1
            if scene_retry_count >= max_retries_per_scene:
                cprint(
                    f"Task: {args.task_id} Episode: {episode_idx} Scene: {scene_idx} reached {max_retries_per_scene} failed attempts, switching scene.",
                    "yellow",
                )
                scene_idx += 1
                scene_retry_count = 0
            continue

        if (not ep_success) or (ep_success_times < args.min_success_steps):
            cprint(
                f"Task: {args.task_id} Episode: {episode_idx} Scene: {scene_idx} failed with reward {ep_reward} and success times {ep_success_times}",
                "red",
            )
            scene_retry_count += 1
            if scene_retry_count >= max_retries_per_scene:
                cprint(
                    f"Task: {args.task_id} Episode: {episode_idx} Scene: {scene_idx} reached {max_retries_per_scene} failed attempts, switching scene.",
                    "yellow",
                )
                scene_idx += 1
                scene_retry_count = 0
            continue

        scene_retry_count = 0
        total_count += total_count_sub
        if args.quiet:
            process_bar.update(1)

        # save visualized data (first frame + full video)
        sample_video_array = np.stack(img_arrays_sub, axis=0)
        save_video_imageio(
            sample_video_array,
            video_dir / f"episode_{episode_idx}.mp4",
            quiet=args.quiet,
        )
        save_rgb_image(
            img_arrays_sub[0],
            image_dir / f"episode_{episode_idx}_rgb.png",
            quiet=args.quiet,
        )
        save_point_cloud_ply(
            point_cloud_arrays_sub[0],
            point_cloud_dir / f"episode_{episode_idx}_point_cloud.ply",
            quiet=args.quiet,
        )
        save_point_cloud_ply(
            point_cloud_no_robot_arrays_sub[0],
            point_cloud_no_robot_dir / f"episode_{episode_idx}_no_robot.ply",
            quiet=args.quiet,
        )

        episode_ends_arrays.append(copy.deepcopy(total_count))
        img_arrays.extend(copy.deepcopy(img_arrays_sub))
        point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
        point_cloud_no_robot_arrays.extend(copy.deepcopy(point_cloud_no_robot_arrays_sub))
        robot_state_arrays.extend(copy.deepcopy(robot_state_arrays_sub))
        raw_state_arrays.extend(copy.deepcopy(raw_state_arrays_sub))
        action_arrays.extend(copy.deepcopy(action_arrays_sub))
        reward_arrays.extend(copy.deepcopy(reward_arrays_sub))
        env_info_arrays.extend(copy.deepcopy(env_info_arrays_sub))
        texts.extend(copy.deepcopy(texts_sub))

        del (
            img_arrays_sub,
            point_cloud_arrays_sub,
            point_cloud_no_robot_arrays_sub,
            robot_state_arrays_sub,
            raw_state_arrays_sub,
            action_arrays_sub,
            reward_arrays_sub,
            env_info_arrays_sub,
            texts_sub,
        )

        if not args.quiet:
            cprint(
                "Episode Index: {}, Episode End: {}, Reward: {}, Success Times: {}".format(
                    episode_idx, total_count, ep_reward, ep_success_times
                ),
                "green",
            )

        episode_idx += 1
        scene_idx += 1

    # Merge data
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.ndim == 4 and img_arrays.shape[1] == 3:
        img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
    if img_arrays.dtype != np.uint8:
        img_arrays = np.clip(img_arrays, 0, 255).astype(np.uint8)

    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0).astype(np.float32)
    point_cloud_no_robot_arrays = np.stack(point_cloud_no_robot_arrays, axis=0).astype(np.float32)
    robot_state_arrays = np.stack(robot_state_arrays, axis=0).astype(np.float32)
    raw_state_arrays = np.stack(raw_state_arrays, axis=0).astype(np.float32)
    action_arrays = np.stack(action_arrays, axis=0).astype(np.float32)
    reward_arrays = np.stack(reward_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)
    texts = np.array(texts, dtype=object)

    # Save data
    Logger.log_info("Saving data to zarr file...", end="", flush=True)
    zarr_dir = pathlib.Path(args.save_dir) / f"{args.task_id}_{args.camera_name}.zarr"
    zarr_root = zarr.group(zarr_dir)
    zarr_data = zarr_root.create_group("data", overwrite=True)
    zarr_meta = zarr_root.create_group("meta", overwrite=True)

    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    robot_state_chunk_size = (100, robot_state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset(
        "images",
        data=img_arrays,
        chunks=img_chunk_size,
        dtype="uint8",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_clouds",
        data=point_cloud_arrays,
        chunks=point_cloud_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_clouds_no_robot",
        data=point_cloud_no_robot_arrays,
        chunks=point_cloud_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "robot_states",
        data=robot_state_arrays,
        chunks=robot_state_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "actions",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends", data=episode_ends_arrays, dtype="int64", compressor=compressor
    )
    zarr_data.create_dataset(
        "texts", data=texts, dtype=object, compressor=compressor, object_codec=MsgPack()
    )

    print("Done")
    Logger.log_info(f"Dataset Info:\n{zarr_root.tree()}")
    Logger.print_seperator()

    custom_split_size = max(1, min(len(episode_ends_arrays), max(10, args.num_episodes // 10)))

    del (
        img_arrays,
        point_cloud_arrays,
        point_cloud_no_robot_arrays,
        robot_state_arrays,
        raw_state_arrays,
        action_arrays,
        reward_arrays,
        episode_ends_arrays,
        texts,
    )
    del zarr_root, zarr_data, zarr_meta
    Logger.log_info("Delete the data in memory")

    dataset = MetaWorldDataset(
        data_dir=zarr_dir,
        split="custom",
        custom_split_size=custom_split_size,
    )
    dataset.print_info()

    Logger.log_ok("All data saved successfully!")


if __name__ == "__main__":
    # ManiSkill uses SAPIEN renderer; default to EGL for headless servers
    os.environ.setdefault("SAPIEN_RENDER_SYSTEM", "egl")

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=str, default="PickCube-v1")
    parser.add_argument("--camera-name", type=str, default="base_camera")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--obs-mode", type=str, default="pointcloud")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "data_new"
            / "maniskill"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--min-success-steps", type=int, default=5)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")

    main(parser.parse_args())
