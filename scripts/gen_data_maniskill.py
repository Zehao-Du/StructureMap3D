import argparse
import copy
import os
import pathlib
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import tqdm
import zarr
from numcodecs import MsgPack
from termcolor import colored, cprint

# 兼容直接脚本运行：
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from MapPolicy.dataset.metaworld_dataset import MetaWorldDataset
from MapPolicy.envs.maniskill_wrapper_env import ManiSkillEnv
from MapPolicy.helpers.Common import (
    save_video_imageio,
)
from MapPolicy.helpers.Logger import Logger


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


def _run_command(cmd: list[str], quiet: bool = False):
    Logger.log_info(f"Running command: {' '.join(cmd)}")
    if quiet:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed with code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
    else:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def _generate_raw_trajectory(args, record_dir: pathlib.Path, traj_name: str) -> pathlib.Path:
    cmd = [
        sys.executable,
        "-m",
        "mani_skill.examples.motionplanning.panda.run",
        "-e",
        args.task_id,
        "-n",
        str(args.num_episodes),
        "--only-count-success",
        "--record-dir",
        str(record_dir),
        "--traj-name",
        traj_name,
        "--num-procs",
        str(args.mp_num_procs),
        "-b",
        "cpu",
    ]
    _run_command(cmd, quiet=args.quiet)
    traj_path = record_dir / args.task_id / "motionplanning" / f"{traj_name}.h5"
    if not traj_path.exists():
        raise FileNotFoundError(f"Raw trajectory not found: {traj_path}")
    return traj_path


def _replay_convert_trajectory(
    raw_traj_path: pathlib.Path,
    target_control_mode: str,
    replay_num_envs: int,
    quiet: bool = False,
) -> pathlib.Path:
    cmd = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(raw_traj_path),
        "--save-traj",
        "--target-control-mode",
        target_control_mode,
        "--obs-mode",
        "none",
        "--num-envs",
        str(replay_num_envs),
        "-b",
        "physx_cpu",
    ]
    _run_command(cmd, quiet=quiet)

    converted_path = raw_traj_path.with_name(
        f"{raw_traj_path.stem}.none.{target_control_mode}.physx_cpu.h5"
    )
    if not converted_path.exists():
        raise FileNotFoundError(f"Converted trajectory not found: {converted_path}")
    return converted_path


def _load_episode_actions(traj_h5_path: pathlib.Path):
    json_path = traj_h5_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Converted trajectory json not found: {json_path}")

    import json

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    episodes = meta.get("episodes", [])
    episode_actions = []
    with h5py.File(traj_h5_path, "r") as h5f:
        for episode in episodes:
            eid = episode["episode_id"]
            traj_key = f"traj_{eid}"
            if traj_key not in h5f:
                continue
            actions = np.asarray(h5f[traj_key]["actions"][:], dtype=np.float32)
            seed = episode.get("episode_seed", None)
            reset_kwargs = episode.get("reset_kwargs", {}) or {}
            success = bool(episode.get("success", False))
            episode_actions.append((seed, reset_kwargs, success, actions))
    return episode_actions


def main(args):
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with arguments:'
    )
    Logger.log_info(f"Task id: {args.task_id}")
    Logger.log_info(f"Camera name: {args.camera_name}")
    Logger.log_info(f"Image size: {args.image_size}")
    Logger.log_info(f"Obs mode: {args.obs_mode}")
    Logger.log_info(f"Control mode (target): {args.control_mode}")
    Logger.log_info(f"Num points: {args.num_points}")
    Logger.log_info(f"Number of episodes: {args.num_episodes}")
    Logger.log_info(f"Episode length: {args.episode_length}")
    Logger.log_info(f"Motion-planning CPU procs: {args.mp_num_procs}")
    Logger.log_info(f"Replay CPU envs: {args.replay_num_envs}")
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
    video_dir.mkdir(parents=True, exist_ok=True)

    if args.control_mode != "pd_ee_delta_pos":
        Logger.log_warning(
            "Current pipeline is designed for replay conversion to pd_ee_delta_pos. "
            f"Received target mode={args.control_mode}."
        )

    raw_record_dir = pathlib.Path(args.save_dir) / "raw_motionplanning"
    raw_record_dir.mkdir(parents=True, exist_ok=True)
    raw_traj_name = f"{args.task_id}_mp_{args.num_episodes}eps"
    raw_traj_path = _generate_raw_trajectory(args, raw_record_dir, raw_traj_name)
    converted_traj_path = _replay_convert_trajectory(
        raw_traj_path,
        target_control_mode=args.control_mode,
        replay_num_envs=args.replay_num_envs,
        quiet=args.quiet,
    )
    converted_episodes = _load_episode_actions(converted_traj_path)
    if len(converted_episodes) == 0:
        raise RuntimeError(f"No converted episodes loaded from {converted_traj_path}")

    Logger.log_info(
        f"Loaded {len(converted_episodes)} converted episodes from {converted_traj_path}"
    )

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

    description = args.text if args.text is not None else args.task_id

    episode_idx = 0
    source_episode_idx = 0
    if args.quiet:
        process_bar = tqdm.tqdm(range(args.num_episodes))

    while episode_idx < args.num_episodes and source_episode_idx < len(converted_episodes):
        seed, reset_kwargs, converted_success, converted_actions = converted_episodes[source_episode_idx]
        source_episode_idx += 1

        if not converted_success:
            cprint(
                f"Task: {args.task_id} Converted episode {source_episode_idx-1} marked unsuccessful in replay metadata, skipping.",
                "yellow",
            )
            continue

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

        reset_seed = reset_kwargs.get("seed", seed)
        reset_options = reset_kwargs.get("options", None)
        try:
            env.reset(seed=int(reset_seed) if reset_seed is not None else None, options=reset_options)
        except Exception:
            env.reset(seed=int(reset_seed) if reset_seed is not None else None)

        for action in converted_actions:
            obs_dict = copy.deepcopy(env.get_obs())
            obs_img = obs_dict["image"]
            obs_robot_state = obs_dict["robot_state"]
            obs_raw_state = obs_dict["raw_state"]
            obs_point_cloud = obs_dict["point_cloud"]
            obs_point_cloud_no_robot = obs_dict["point_cloud_no_robot"]

            _, reward, terminated, truncated, env_info = env.step(action)

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

            # 不提前中断：与 ManiSkill replay_trajectory 对齐，完整执行转换后的动作序列。
            # 否则会丢掉轨迹末尾若干步（常见于已 success 后的尾段动作）。

        total_count_sub = len(action_arrays_sub)

        if total_count_sub == 0:
            cprint(
                f"Task: {args.task_id} Episode: {episode_idx} failed: no trajectory recorded",
                "red",
            )
            continue

        if not ep_success:
            cprint(
                f"Task: {args.task_id} Episode: {episode_idx} rollout success_once=False, but keeping episode because converted replay metadata is successful.",
                "yellow",
            )

        total_count += total_count_sub
        if args.quiet:
            process_bar.update(1)

        # save visualized data (full video only)
        sample_video_array = np.stack(img_arrays_sub, axis=0)
        save_video_imageio(
            sample_video_array,
            video_dir / f"episode_{episode_idx}.mp4",
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

    if episode_idx < args.num_episodes:
        Logger.log_warning(
            f"Only collected {episode_idx} successful episodes after conversion, fewer than requested {args.num_episodes}."
        )

    if len(img_arrays) == 0:
        raise RuntimeError("No successful episodes collected; zarr will not be written.")

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
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--mp-num-procs", type=int, default=1)
    parser.add_argument("--replay-num-envs", type=int, default=1)
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
    parser.add_argument("--quiet", action="store_true")

    main(parser.parse_args())
