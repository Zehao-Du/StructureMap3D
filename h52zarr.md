生成原始.h5数据

python -m mani_skill.examples.motionplanning.panda.run --env-id "PickCube-v1" --num-traj 10 --only-count-success --save-video --record-dir /data2/lirui/mani --traj-name test

replay数据并获取点云，更改控制方式

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /data2/lirui/mani/PickCube-v1/motionplanning/…….h5目录 \
  -c pd_ee_delta_pos \
  -o pointcloud \
  --no-vis \
  --save-traj \
  --use-env-states 


转化为zarr格式
h52zarr.py