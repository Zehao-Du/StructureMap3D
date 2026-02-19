import zarr
import numpy as np

# 1. 加载Zarr数据集（支持本地文件路径/文件夹，或云存储路径）
# 方式1：加载整个Zarr存储（文件夹形式）
zarr_store = zarr.open("/data2/lirui/StructureMap3D/data_new/maniskill/PegInsertionSide-v1_base_camera.zarr", mode="r")  # r=只读，避免修改数据

# 2. 查看数据结构（关键：确认维度、形状、数据类型）
print("=== Zarr 数据结构 ===")
# 若为Zarr Group（包含多个数组）
if isinstance(zarr_store, zarr.Group):
    print("所有数组名称：", list(zarr_store.keys()))  # 比如['states', 'actions', 'rewards']
    # 查看其中一个数组（比如actions）
    actions = zarr_store["data"]
else:
    actions = zarr_store  # 若为单个数组

# 打印核心信息
print("动作数组形状：", actions.shape)  # 比如(10000, 2) → (样本数, 动作维度)
print("动作数据类型：", actions.dtype)  # 比如float32（需确认是合理类型）
print("动作数组维度名（若有）：", actions.attrs.get("dimensions", "无"))
print("动作数组范围：", np.min(actions[:1000]), "~", np.max(actions[:1000])) 