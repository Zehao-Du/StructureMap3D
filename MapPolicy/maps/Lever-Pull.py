"""
Lever-Pull: 3D Structure Map for the Lever-Pull manipulation task.

Lever_Mechanism 四部件：
1. 灰色长方体主体 (Cuboid)
2. 插在灰色主体上的深灰色细圆柱 (Cylinder)
3. 连接在细圆柱末端、与细圆柱垂直的深灰色细圆柱2号 (Cylinder，绕 X 转 90° 使轴线沿 Z)
4. 连接在细圆柱2号末端的球 (Sphere)

Edges:
- 球 — 圆柱2: Fixed（球的外侧面 — 圆柱2顶面）
- 圆柱2 — 圆柱1: Planar-Contact（圆柱2底面 — 圆柱1侧面）
- 球 — 圆柱1: Revolute（球穿过中心、与圆柱1轴线平行的轴绕圆柱1轴线旋转）
- 圆柱1 — 主体: Fixed（圆柱1与主体刚性连接）
- 其余节点对: Free（由 _Add_Free_Edge 补全）
"""

import torch
from Structure_Primitive import Cuboid, Cylinder, Sphere
from base_template import StructureEdge, StructureGraph


class Lever_Mechanism:
    """
    四部件：灰色长方体主体、深灰细圆柱1（与主体 Fixed）、深灰细圆柱2（底面与圆柱1侧面 Planar-Contact）、
    球（与圆柱2 Fixed，绕圆柱1轴线 Revolute）。
    """

    def __init__(self, sizes, positions, rotations):
        # sizes: [B, 8]  ->  cuboid(3) + cyl1(2: height, top_radius) + cyl2(2) + sphere(1: radius)
        # positions: [B, 12],  rotations: [B, 24]

        size_cuboid = sizes[:, 0:3]
        size_cyl1 = sizes[:, 3:5]
        size_cyl2 = sizes[:, 5:7]
        size_sphere = sizes[:, 7:8]

        pos_cuboid = positions[:, 0:3]
        pos_cyl1 = positions[:, 3:6]
        pos_cyl2 = positions[:, 6:9]
        pos_sphere = positions[:, 9:12]

        rot_cuboid = rotations[:, 0:6]
        rot_cyl1 = rotations[:, 6:12]
        rot_cyl2 = rotations[:, 12:18]
        rot_sphere = rotations[:, 18:24]

        Nodes = []

        Nodes.append(
            Cuboid(
                size_cuboid[:, 0],
                size_cuboid[:, 1],
                size_cuboid[:, 2],
                position=pos_cuboid,
                rotation=rot_cuboid,
                Semantic="lever base body",
            )
        )
        Nodes.append(
            Cylinder(
                size_cyl1[:, 0],
                size_cyl1[:, 1],
                position=pos_cyl1,
                rotation=rot_cyl1,
                Semantic="lever cylinder 1",
            )
        )
        Nodes.append(
            Cylinder(
                size_cyl2[:, 0],
                size_cyl2[:, 1],
                position=pos_cyl2,
                rotation=rot_cyl2,
                Semantic="lever cylinder 2",
            )
        )
        Nodes.append(
            Sphere(
                size_sphere[:, 0],
                position=pos_sphere,
                rotation=rot_sphere,
                Semantic="lever handle sphere",
            )
        )

        Edges = []
        # 0=主体(Cuboid), 1=圆柱1(Cylinder), 2=圆柱2(Cylinder), 3=球(Sphere)
        # 根据文档，Planar-Contact 约束需要 Face $i$ + Face $j$，要求两个面平行且接触
        # Cylinder Faces: 0=Top, 1=Bottom, 2=Side
        # Sphere Faces: 0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-
        # Cuboid Faces: 0=Back, 1=Front, 2=Top, 3=Bottom, 4=Left, 5=Right
        
        # 球 — 圆柱2: Fixed（球的外侧面 — 圆柱2 顶面 Face 0）
        # 使用球的 +Z 方向面（Face 4）与圆柱2的顶面
        Edges.append(
            StructureEdge(3, 2, "Fixed", {"type": 0, "idx": 4}, {"type": 0, "idx": 0}, [0, 0, 0])
        )
        
        # 圆柱2 — 圆柱1: Planar-Contact（圆柱2 底面 Face 1 — 圆柱1 侧面 Face 2）
        Edges.append(
            StructureEdge(2, 1, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 2}, [0, 0, 0])
        )
        
        # 球 — 圆柱1: Revolute（球穿过中心、与圆柱1轴线平行的轴绕圆柱1轴线旋转）
        # Revolute 要求两轴平行：圆柱1 Axis 0（沿 Y）↔ 球 Axis 1（Y 方向，过球心）
        Edges.append(
            StructureEdge(3, 1, "Revolute", {"type": 1, "idx": 1}, {"type": 1, "idx": 0}, [0, 0, 0])
        )
        
        # 圆柱1 — 主体: Fixed（圆柱1与主体刚性连接）
        # Cuboid Faces: 2=Top, 3=Bottom; Cylinder Faces: 0=Top, 1=Bottom
        # 圆柱1底面贴合主体顶面
        Edges.append(
            StructureEdge(1, 0, "Fixed", {"type": 0, "idx": 1}, {"type": 0, "idx": 2}, [0, 0, 0])
        )

        self.Nodes = Nodes
        self.Edges = Edges


class StructureMap_LeverPull(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        :param sizes: [B, 8]  (cuboid [3] + cyl1 [2] + cyl2 [2] + sphere [1])
        :param positions: [B, 12]
        :param rotations: [B, 24]  (4 * 6D)
        Node: 4。Edges: 球–圆柱2 Fixed，圆柱2–圆柱1 Planar-Contact，球–圆柱1 Revolute，圆柱1–主体 Fixed；其余 Free。
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = [Lever_Mechanism(sizes, positions, rotations)]

        Nodes = []
        Edges = []
        num_node = 0
        for obj in Objects:
            for node in obj.Nodes:
                Nodes.append(node)
            for edge in obj.Edges:
                edge.update_node_idx(num_node)
                Edges.append(edge)
            num_node += len(obj.Nodes)

        super().__init__(Nodes, Edges, clip_model)

    def _preprocess_parameters(self, sizes):
        size_range = (0.02, 5)
        min_s, max_s = size_range
        sizes = torch.sigmoid(sizes) * (max_s - min_s) + min_s
        return sizes


if __name__ == "__main__":
    import os

    os.environ["WAYLAND_DISPLAY"] = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 1

    print(f"Initializing Lever-Pull (4 parts) on {device}...")

    # ==========================================
    # 1. 尺寸 (Y 为上)  Cuboid(h,l,w); Cylinder(height, top_radius); Sphere(radius)
    # ==========================================

    # 灰色长方体主体
    h_c, l_c, w_c = 0.12, 0.15, 0.18
    size_cuboid = [h_c, l_c, w_c]

    # 深灰细圆柱1：插在主体上，轴线沿 Y
    H1, r1 = 0.12, 0.015
    size_cyl1 = [H1, r1]

    # 深灰细圆柱2：在圆柱1末端，与圆柱1垂直，轴线沿 Z（需绕 X 转 90°）
    H2, r2 = 0.10, 0.012
    size_cyl2 = [H2, r2]

    # 球：在圆柱2末端
    R = 0.025
    size_sphere = [R]

    sizes_list = size_cuboid + size_cyl1 + size_cyl2 + size_sphere
    sizes = torch.tensor([sizes_list], dtype=torch.float32, device=device)

    # ==========================================
    # 2. 旋转  圆柱2 绕 X 转 90° 使轴线从 Y 变为 Z
    # ==========================================
    identity_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    rot_x90_6d = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    rotations = torch.tensor(
        [identity_6d + identity_6d + rot_x90_6d + identity_6d],
        dtype=torch.float32,
        device=device,
    )

    # ==========================================
    # 3. 位置  主体中心 y=h_c/2；圆柱1插在顶面中心，半入半出；圆柱2在圆柱1顶端；球在圆柱2的+Z端
    # ==========================================
    y_c = h_c / 2.0
    pos_cuboid = [0.0, y_c, 0.0]

    # 圆柱1：中心在主体顶面 y=h_c，半截插入
    pos_cyl1 = [0.0, h_c+H1/2, 0]

    # 圆柱2：在圆柱1顶端 y = h_c + H1/2，中心在 (0, y, 0)，沿 Z 向 ±H2/2
    y_cyl2 = h_c+H1+r2
    pos_cyl2 = [0.0, y_cyl2, 0.04]

    # 球：在圆柱2的 +Z 端，与柱面相切
    pos_sphere = [0.0, y_cyl2, (H2 / 2.0) + R]

    positions_list = pos_cuboid + pos_cyl1 + pos_cyl2 + pos_sphere
    positions = torch.tensor([positions_list], dtype=torch.float32, device=device)

    # ==========================================
    # 4. 初始化与可视化
    # ==========================================
    from utils import CLIPEncoder

    try:
        clip_encoder = CLIPEncoder("ViT-B/32").to(device)
        graph = StructureMap_LeverPull(sizes, positions, rotations, clip_encoder)

        graph.Node[0].visual_color = [0.5, 0.5, 0.5]
        graph.Node[1].visual_color = [0.35, 0.35, 0.38]
        graph.Node[2].visual_color = [0.35, 0.35, 0.38]
        graph.Node[3].visual_color = [0,0,1]

        print(f"Graph: {len(graph.Node)} Nodes, {len(graph.Edge)} Edges")
        print("Visualizing Lever-Pull (4 parts: Fixed + Revolute + Planar-Contact + Free)...")

        from visualization_helper import visualize_structure_graph

        visualize_structure_graph(graph)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
