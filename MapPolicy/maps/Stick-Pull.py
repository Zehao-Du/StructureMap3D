"""
Stick-Pull: 3D Structure Map for the Stick Pull manipulation task.

Based on Metaworld's SawyerStickPullEnvV3:
- stick: 细长圆柱 (Cylinder)，轴向 X，穿过把手孔。
- thermos: 保温杯 base/body/handle。底座与杯体 Fixed；把手外侧面与杯体侧面 Planar-Contact，顶面开洞的棱与杯体轴线 Alignment；杆与其他部件 Free。

四部件:
1. 杆子 Stick (Cylinder) — 细长圆柱，轴向 X，穿过孔
2. 保温杯底座 (Cylinder)
3. 保温杯瓶身 (Cylinder)
4. 保温杯把手 (Rectangular_Ring) — 孔轴 X，一侧紧贴杯身

Edges:
- 底座—杯体: Fixed（底座顶面 — 杯体底面，两圆柱刚性连接）
- 把手—杯体: Planar-Contact（把手外侧面 — 杯体侧面）+ Alignment（顶面开洞的棱 ‖ 杯体轴线）
- 其他节点对: Free（由 _Add_Free_Edge 自动添加）
"""

import torch
from Structure_Primitive import Cylinder, Rectangular_Ring
from base_template import StructureEdge, StructureGraph


class Stick_Pull_Mechanism:
    """
    四部件：杆(细长圆柱)、保温杯底座、瓶身、把手(Rectangular_Ring，孔轴 X，贴杯壁)。
    Edges: 底座—杯体 Fixed；把手—杯体 Planar-Contact + Alignment（顶面开洞棱‖杯体轴线）；其他 Free。
    """

    def __init__(self, sizes, positions, rotations):
        # sizes: [B, 11]  ->  stick(2: height, radius) + base(2) + body(2) + handle(5)
        # positions: [B, 12],  rotations: [B, 24]

        size_stick = sizes[:, 0:2]
        size_base = sizes[:, 2:4]
        size_body = sizes[:, 4:6]
        size_handle = sizes[:, 6:11]

        pos_stick = positions[:, 0:3]
        pos_base = positions[:, 3:6]
        pos_body = positions[:, 6:9]
        pos_handle = positions[:, 9:12]

        rot_stick = rotations[:, 0:6]
        rot_base = rotations[:, 6:12]
        rot_body = rotations[:, 12:18]
        rot_handle = rotations[:, 18:24]

        Nodes = []

        # 杆：细长圆柱，Cylinder 默认轴向 Y，需 rot 使轴向 X 以穿过把手孔
        Nodes.append(
            Cylinder(
                size_stick[:, 0],   # height: 沿轴向长度
                size_stick[:, 1],   # top_radius (= bottom_radius)
                position=pos_stick,
                rotation=rot_stick,
                Semantic="stick",
            )
        )
        Nodes.append(
            Cylinder(
                size_base[:, 0],
                size_base[:, 1],
                position=pos_base,
                rotation=rot_base,
                Semantic="thermos base",
            )
        )
        Nodes.append(
            Cylinder(
                size_body[:, 0],
                size_body[:, 1],
                position=pos_body,
                rotation=rot_body,
                Semantic="thermos body",
            )
        )
        # Rectangular_Ring: 孔轴(local Y)经 rot_z_neg90 对齐 world X；local -X 面贴杯壁
        # 参数: front_height(孔沿X的延伸), outer_L(径向,贴杯侧到外缘), outer_W(Z向), inner_L, inner_W
        Nodes.append(
            Rectangular_Ring(
                size_handle[:, 0],   # front_height: 孔沿 X 的延伸
                size_handle[:, 1],   # outer_top_length: 外框 径向(rot 后=Y 向，贴杯侧到外缘)
                size_handle[:, 2],   # outer_top_width: 外框 Z 向
                size_handle[:, 3],   # inner_top_length: 孔 X 向
                size_handle[:, 4],   # inner_top_width: 孔 Z 向
                position=pos_handle,
                rotation=rot_handle,
                Semantic="thermos handle",
            )
        )

        self.Nodes = Nodes
        # 根据文档，Planar-Contact 约束需要 Face $i$ + Face $j$，要求两个面平行且接触
        # 0=杆(Cylinder), 1=底座(Cylinder), 2=杯体(Cylinder), 3=把手(Rectangular_Ring)
        # Cylinder Faces: 0=Top, 1=Bottom, 2=Side
        # Rectangular_Ring Faces: 0-3=外侧面, 4-7=内侧面, 8-11=上底, 12-15=下底
        
        Edges = []
        
        # 底座—杯体: Fixed（底座顶面 Face 0 — 杯体底面 Face 1，两圆柱刚性连接）
        Edges.append(
            StructureEdge(1, 2, "Fixed", {"type": 0, "idx": 0}, {"type": 0, "idx": 1}, [0, 0, 0])
        )
        
        # 把手—杯体: Planar-Contact（把手外侧面 Face 1 — 杯体侧面 Face 2）
        Edges.append(
            StructureEdge(3, 2, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 2}, [0, 0, 0])
        )
        
        # 把手—杯体: Alignment（顶面开洞的棱 ‖ 大圆柱轴线）
        # Rectangular_Ring Axis 13~16: 顶面内孔棱（Top Inner Edges）；Cylinder Axis 0: 中心轴
        Edges.append(
            StructureEdge(3, 2, "Alignment", {"type": 1, "idx": 14}, {"type": 1, "idx": 0}, [0, 0, 0])
        )
        
        # 注意：其他节点对之间没有显式约束，将由 _Add_Free_Edge 自动添加 Free 边
        # 包括：杆子(0)与底座(1)、杆子(0)与杯体(2)、杆子(0)与把手(3)之间都是 Free
        
        self.Edges = Edges


class StructureMap_StickPull(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        :param sizes: [B, 11]  (stick [2: height, radius] + base [2] + body [2] + handle [5])
        :param positions: [B, 12]
        :param rotations: [B, 24]
        Node: 4。Edges: 底座–杯体 Fixed，把手–杯体 Planar-Contact + Alignment；其余 Free。
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = [Stick_Pull_Mechanism(sizes, positions, rotations)]

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

    print(f"Initializing Stick-Pull (4 parts: base-body Fixed, handle-body Planar-Contact+Alignment, stick Free) on {device}...")

    # 杆: 细长圆柱 (Cylinder)，height=沿轴向长度，radius=半径；轴向需 rot 到 X 以穿过孔
    h_stick, r_stick = 0.12, 0.01   # 12cm 长、1cm 半径，细长条
    size_stick = [h_stick, r_stick]

    # 保温杯: base cyl 0.062 0.02, body 0.06 0.1; 把手 Rectangular_Ring 一侧紧贴杯身侧壁
    H_base, r_base = 0.04, 0.062
    H_body, r_body = 0.2, 0.06
    # handle: front_height(孔沿X的延伸), outer_L(径向深度), outer_W(Z向), inner_L, inner_W(孔)
    # 孔要很大、易见：inner 接近 outer，环框较薄
    h_ring, outer_L, outer_W = 0.068, 0.05, 0.08   # 外框 5cm x 8cm
    inner_L, inner_W = 0.04, 0.065                 # 孔 4cm x 6.5cm，框厚约 5~7.5mm

    sizes_list = size_stick + [H_base, r_base] + [H_body, r_body] + [h_ring, outer_L, outer_W, inner_L, inner_W]
    sizes = torch.tensor([sizes_list], dtype=torch.float32, device=device)

    identity_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # 圆柱默认轴向 Y；绕 X 转 -90° 使轴向变为 Z，底座在下、瓶身在上沿 Z 叠放贴紧
    rot_x_neg90_6d = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # 绕 Z 转 -90°: local Y->world X, local X->world -Y
    rot_z_neg90_6d = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
    # 杆 Cylinder 默认轴向 Y；rot_z_neg90 使轴向变为 X，与把手孔轴一致以便穿过
    # 把手 Ring: rot_z_neg90 使孔轴(local Y)->world X
    rotations = torch.tensor(
        [rot_z_neg90_6d + rot_x_neg90_6d + rot_x_neg90_6d + rot_z_neg90_6d],
        dtype=torch.float32,
        device=device,
    )

    # thermos 在 (tx,ty)，底座底 z=0，瓶身叠在底座上；把手孔沿 X，贴杯壁
    # 杆穿过把手孔：与把手同 y、z，即 pos_stick = [tx, ty - r_body - outer_L/2, H_base + 0.07]
    tx, ty = 0.2, 0.6
    pos_stick = [tx+0.8, ty - r_body - outer_L / 2.0, H_base + 0.07]
    pos_base = [tx, ty, H_base / 2.0]              # 底座中心 z=0.02，柱体 z∈[0, H_base]，顶面 z=H_base
    pos_body = [tx, ty, H_base + H_body / 2.0]     # 瓶身中心 z=0.14，柱体 z∈[H_base, H_base+H_body]，底面 z=H_base 与底座顶贴紧
    pos_handle = [tx, ty - r_body - outer_L / 2.0, H_base + 0.07]  # -X 面贴杯壁 y=ty-r_body

    positions = torch.tensor(
        [pos_stick + pos_base + pos_body + pos_handle],
        dtype=torch.float32,
        device=device,
    )

    from utils import CLIPEncoder

    try:
        clip_encoder = CLIPEncoder("ViT-B/32").to(device)
        graph = StructureMap_StickPull(sizes, positions, rotations, clip_encoder)

        graph.Node[0].visual_color = [0.2, 0.4, 0.9]
        graph.Node[1].visual_color = [0.5, 0.5, 0.55]
        graph.Node[2].visual_color = [0.3, 0.6, 0.35]
        graph.Node[3].visual_color = [0.85, 0.5, 0.2]   # 把手橙色，便于找到孔

        print(f"Graph: {len(graph.Node)} Nodes, {len(graph.Edge)} Edges")
        from visualization_helper import visualize_structure_graph

        visualize_structure_graph(graph)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
