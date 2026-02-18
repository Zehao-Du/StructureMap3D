import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from MapPolicy.envs.metaworld_env import MetaWorldEnv
from MapPolicy.helpers.mujoco import camera_name_to_id
from MapPolicy.helpers.Common import set_seed
from MapPolicy.helpers.graphics import HomogeneousCoordinates
from scipy.spatial.transform import Rotation


def get_camera_intrinsics(renderer, camera_name):
    """Calculate camera intrinsics for a given camera"""
    mujoco_model = renderer.model
    width, height = renderer.width, renderer.height
    
    camera_id = camera_name_to_id(mujoco_model, camera_name)
    
    aspect_ratio = width / height
    fovy = np.radians(mujoco_model.cam_fovy[camera_id])
    fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)
    fx, fy = width / (2 * np.tan(fovx / 2)), height / (2 * np.tan(fovy / 2))
    cx, cy = width / 2, height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return {
        'camera_name': camera_name,
        'width': width,
        'height': height,
        'fovy': np.degrees(fovy),
        'fovx': np.degrees(fovx),
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'K': K
    }


def get_camera_extrinsics(renderer, camera_name):
    """Calculate camera extrinsics (world to camera 4x4 matrix) for a given camera"""
    mujoco_model = renderer.model
    camera_id = camera_name_to_id(mujoco_model, camera_name)
    
    cam_body_id = mujoco_model.cam_bodyid[camera_id]
    cam_pos = mujoco_model.body_pos[cam_body_id]
    
    c2b_r = np.array(mujoco_model.cam_mat0[camera_id]).reshape((3, 3))
    b2w_r = Rotation.from_quat([0, 1, 0, 0], scalar_first=True).as_matrix()
    c2w_r = np.matmul(c2b_r, b2w_r)
    
    c2w = HomogeneousCoordinates.pos_rot_to_matrix(cam_pos, c2w_r)
    w2c = np.linalg.inv(c2w)
    
    return {
        'camera_name': camera_name,
        'cam_pos': cam_pos,
        'c2b_r': c2b_r,
        'b2w_r': b2w_r,
        'c2w_r': c2w_r,
        'c2w': c2w,
        'w2c': w2c
    }


def main():
    os.environ['MUJOCO_GL'] = "egl"
    set_seed(0)
    
    print("Initializing MetaWorld environment...")
    env = MetaWorldEnv(
        task_name="reach",
        max_episode_length=10,
        image_size=224,
        camera_name="corner",
        use_point_crop=True,
        num_points=1024,
    )
    
    env.reset()
    renderer = env.env.mujoco_renderer
    
    print("\n" + "="*80)
    print("Camera Intrinsics and Extrinsics for MetaWorld")
    print("="*80)
    
    for camera_name in ['corner', 'corner2']:
        intrinsics = get_camera_intrinsics(renderer, camera_name)
        extrinsics = get_camera_extrinsics(renderer, camera_name)
        
        print(f"\n--- Camera: {camera_name} ---")
        print(f"\nIntrinsics:")
        print(f"Resolution: {intrinsics['width']} x {intrinsics['height']}")
        print(f"FOV (vertical): {intrinsics['fovy']:.2f} degrees")
        print(f"FOV (horizontal): {intrinsics['fovx']:.2f} degrees")
        print(f"fx: {intrinsics['fx']:.4f}")
        print(f"fy: {intrinsics['fy']:.4f}")
        print(f"cx: {intrinsics['cx']:.4f}")
        print(f"cy: {intrinsics['cy']:.4f}")
        print("\nIntrinsic Matrix (K) [3x3]:")
        print(intrinsics['K'])
        
        print(f"\nExtrinsics:")
        print(f"Camera Position: {extrinsics['cam_pos']}")
        print(f"\nCamera to World 4x4 Matrix (c2w):")
        print(extrinsics['c2w'])
        print(f"\nWorld to Camera 4x4 Matrix (w2c) - this is what's used in frust_masked_chamfer_loss:")
        print(extrinsics['w2c'])
        print(f"\nCamera to World Rotation (c2w_r) [3x3] - this is what's currently in point_cloud_complementation.py:")
        print(extrinsics['c2w_r'])
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
