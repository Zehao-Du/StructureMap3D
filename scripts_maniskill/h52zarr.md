生成原始.h5数据

python -m mani_skill.examples.motionplanning.panda.run --env-id "StackCube-v1" --num-traj 1000 --only-count-success --save-video --record-dir maniskill/data --traj-name StackCube

replay数据并获取点云，更改控制方式

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /data2/lirui/maniskill/data/StackCube-v1/motionplanning/StackCube.h5 \
  -c pd_ee_delta_pose \
  -o rgbd \
  --no-vis \
  --save-traj \
  --verbose \
  --no-allow-failure


转化为zarr格式
python h52zarr.py --input-path /data2/lirui/StructureMap3D/maniskill/data/PickCube-v1/motionplanning/PickCube.pointcloud.pd_ee_delta_pose.physx_cpu.h5 --zarr-save-dir /data2/lirui/StructureMap3D/data_new/maniskill_zarr/PickCube-v1