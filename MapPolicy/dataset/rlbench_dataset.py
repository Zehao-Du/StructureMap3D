import torch
import zarr
from termcolor import colored

from MapPolicy.helpers.Logger import Logger


class RLBenchDataset(torch.utils.data.Dataset):
    """
    Dataset for RLBench Benchmark.
    Compatible with MapPolicy train loop: returns (image, point_cloud, point_cloud_no_robot, robot_state, raw_state, action, text).
    RLBench zarr does not have point_clouds_no_robot; we use point_cloud for both (no robot masking).
    When num_episodes < 120, uses 80% train / 20% validation instead of fixed 100/20.
    """

    SPLIT_SIZE = {"train": 100, "validation": 20, "custom": None}

    def __init__(self, data_dir, split: str = None, custom_split_size: int = None):
        zarr_root = zarr.open_group(data_dir, mode="r")
        self._episode_ends = zarr_root["meta"]["episode_ends"][:]
        num_episodes = len(self._episode_ends)

        if split not in self.SPLIT_SIZE:
            raise ValueError(f"Invalid split: {split}")

        if split == "custom" and custom_split_size is None:
            raise ValueError(f"custom_split_size must be provided for split: {split}")

        # When fewer than 120 episodes, use proportional split (80% train, 20% val) so validation is non-empty
        # episode_ends[i] = cumulative frame count after (i+1) episodes
        need_val_end = self.SPLIT_SIZE["train"] + self.SPLIT_SIZE["validation"]  # 120
        total_frames = int(self._episode_ends[-1]) if num_episodes > 0 else 0

        if num_episodes < 2:
            # 1 or 0 episodes: split by frames (90% train, 10% val) so both non-empty
            split_frame = int(0.9 * total_frames) if total_frames >= 2 else max(0, total_frames - 1)
            begin_index, end_index = (
                (0, split_frame) if split == "train" else (split_frame, total_frames)
            ) if split != "custom" else (0, self._episode_ends[custom_split_size - 1])
        elif num_episodes < need_val_end:
            train_size = max(1, int(0.8 * num_episodes))
            val_size = num_episodes - train_size
            if val_size < 1:
                train_size = num_episodes - 1
                val_size = 1
            train_end_idx = train_size - 1
            val_end_idx = train_size + val_size - 1
            begin_index, end_index = (
                (0, self._episode_ends[train_end_idx])
                if split == "train"
                else (
                    (
                        self._episode_ends[train_end_idx],
                        self._episode_ends[val_end_idx],
                    )
                    if split == "validation"
                    else (0, self._episode_ends[custom_split_size - 1])
                )
            )
        else:
            train_end_idx = self.SPLIT_SIZE["train"] - 1
            val_end_idx = need_val_end - 1
            begin_index, end_index = (
                (0, self._episode_ends[train_end_idx])
                if split == "train"
                else (
                    (
                        self._episode_ends[train_end_idx],
                        self._episode_ends[val_end_idx],
                    )
                    if split == "validation"
                    else (0, self._episode_ends[custom_split_size - 1])
                )
            )

        # (T, H, W, C) -> (T, C, H, W)
        self._images = zarr_root["data"]["images"][begin_index:end_index].transpose(
            0, 3, 1, 2
        )
        assert self._images.shape[1] == 3
        self._point_clouds = zarr_root["data"]["point_clouds"][begin_index:end_index]
        # RLBench zarr has no point_clouds_no_robot; use same as point_clouds
        self._point_clouds_no_robot = self._point_clouds
        self._robot_states = zarr_root["data"]["robot_states"][begin_index:end_index]
        self._actions = zarr_root["data"]["actions"][begin_index:end_index]
        self._texts = zarr_root["data"]["texts"][begin_index:end_index]
        assert len(self._images) == len(self._robot_states) == len(self._actions)
        self._dataset_size = len(self._actions)

    def __getitem__(self, idx):
        image = torch.from_numpy(self._images[idx]).float()
        point_cloud = torch.from_numpy(self._point_clouds[idx]).float()
        point_cloud_no_robot = torch.from_numpy(self._point_clouds_no_robot[idx]).float()
        robot_state = torch.from_numpy(self._robot_states[idx]).float()
        action = torch.from_numpy(self._actions[idx]).float()
        text = self._texts[idx]
        return image, point_cloud, point_cloud_no_robot, robot_state, torch.zeros((0,)), action, text

    def __len__(self):
        return self._dataset_size

    def print_info(self):
        Logger.log_info(f"RLBench Dataset Info:")
        Logger.log_info(
            f'images ({colored(self._images.dtype, "red")}): {colored(self._images.shape, "red")}, range: [{colored(self._images.min(), "red")}, {colored(self._images.max(), "red")}]'
        )
        Logger.log_info(
            f'point_cloud ({colored(self._point_clouds.dtype, "red")}): {colored(self._point_clouds.shape, "red")}, range: [{colored(self._point_clouds.min(), "red")}, {colored(self._point_clouds.max(), "red")}]'
        )
        Logger.log_info(
            f'robot_state ({colored(self._robot_states.dtype, "red")}): {colored(self._robot_states.shape, "red")}, range: [{colored(self._robot_states.min(), "red")}, {colored(self._robot_states.max(), "red")}]'
        )
        Logger.log_info(
            f'action ({colored(self._actions.dtype, "red")}): {colored(self._actions.shape, "red")}, range: [{colored(self._actions.min(), "red")}, {colored(self._actions.max(), "red")}]'
        )
        Logger.log_info(
            f'episode_ends ({colored(self._episode_ends.dtype, "red")}): {colored(self._episode_ends.shape, "red")}, range: [{colored(self._episode_ends.min(), "red")}, {colored(self._episode_ends.max(), "red")}]'
        )
        Logger.print_seperator()


if __name__ == "__main__":
    data_dir = "data/rlbench/close_box.zarr"
    dataset = RLBenchDataset(data_dir, split="custom", custom_split_size=120)
    dataset.print_info()
    Logger.log_info(len(dataset))
