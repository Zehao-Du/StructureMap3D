import gymnasium as gym
import mani_skill.envs
import numpy as np
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="rgbd",
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode=None,
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism

rgb = obs["sensor_data"]["base_camera"]["rgb"][0]
rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
if rgb.dtype != np.uint8:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

print("RGB stats ::", "shape=", rgb.shape, "dtype=", rgb.dtype, "min=", int(rgb.min()), "max=", int(rgb.max()))

# save rgb to png (headless)
out_dir = pathlib.Path(__file__).resolve().parent / "_outputs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rgb.png"
plt.imsave(out_path, rgb)
print("Saved RGB PNG to:", str(out_path))

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()