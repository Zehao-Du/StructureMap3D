"""
Soccer: 3D Structure Map for the Soccer manipulation task.

Based on Metaworld's SawyerSoccerEnvV3:
- sawyer_soccer.xml: soccer_ball (free) + goal_whole (soccer_goal).
- soccer_goal.xml: 球门 frame + net。
- 球门几何：左柱、右柱沿 X 分立竖直（Z 向），横梁沿 X 架于两柱顶端，球网（薄长方体板）直接贴在横梁后面。

五部件：球 Free；左柱–横梁、右柱–横梁、球网–横梁 Planar-Contact；两门柱轴线 Alignment；球网顶面边–横梁轴线 Alignment。
1. 足球 (Sphere)
2. 球门左柱 (Cuboid)
3. 球门右柱 (Cuboid)
4. 球门横梁 (Cuboid)
5. 球网 (Cuboid，薄板)
"""

import torch
from Structure_Primitive import Cuboid, Sphere
from base_template import StructureEdge, StructureGraph


class Soccer_Mechanism:
    """
    五部件：球、左柱、右柱、横梁、球网（薄板）。
    - 左柱—横梁、右柱—横梁: Planar-Contact；球网一面—横梁后面: Planar-Contact；两门柱轴线平行；球网顶面边‖横梁轴线；球与球门: Free。
    """

    def __init__(self, sizes, positions, rotations):
        # sizes: [B, 13]  ->  ball(1) + post_L(3) + post_R(3) + bar(3) + net(3)
        # positions: [B, 15],  rotations: [B, 30]

        size_ball = sizes[:, 0:1]
        size_post_L = sizes[:, 1:4]
        size_post_R = sizes[:, 4:7]
        size_bar = sizes[:, 7:10]
        size_net = sizes[:, 9:12]

        pos_ball = positions[:, 0:3]
        pos_post_L = positions[:, 3:6]
        pos_post_R = positions[:, 6:9]
        pos_bar = positions[:, 9:12]
        pos_net = positions[:, 12:15]

        rot_ball = rotations[:, 0:6]
        rot_post_L = rotations[:, 6:12]
        rot_post_R = rotations[:, 12:18]
        rot_bar = rotations[:, 18:24]
        rot_net = rotations[:, 24:30]

        Nodes = []

        Nodes.append(
            Sphere(
                size_ball[:, 0],
                position=pos_ball,
                rotation=rot_ball,
                Semantic="soccer ball",
            )
        )
        Nodes.append(
            Cuboid(
                size_post_L[:, 0],
                size_post_L[:, 1],
                size_post_L[:, 2],
                position=pos_post_L,
                rotation=rot_post_L,
                Semantic="goal post left",
            )
        )
        Nodes.append(
            Cuboid(
                size_post_R[:, 0],
                size_post_R[:, 1],
                size_post_R[:, 2],
                position=pos_post_R,
                rotation=rot_post_R,
                Semantic="goal post right",
            )
        )
        Nodes.append(
            Cuboid(
                size_bar[:, 0],
                size_bar[:, 1],
                size_bar[:, 2],
                position=pos_bar,
                rotation=rot_bar,
                Semantic="goal crossbar",
            )
        )
        Nodes.append(
            Cuboid(
                size_net[:, 0],
                size_net[:, 1],
                size_net[:, 2],
                position=pos_net,
                rotation=rot_net,
                Semantic="goal net",
            )
        )

        self.Nodes = Nodes
        Edges = []
        # 0=球, 1=左柱, 2=右柱, 3=横梁, 4=球网
        # Cuboid Faces: 0=Back, 1=Front, 2=Top, 3=Bottom, 4=Left, 5=Right
        
        # 左柱 (1) — 横梁 (3): Planar-Contact（柱顶 Face 2 — 横梁前面 Face 1）
        Edges.append(
            StructureEdge(1, 3, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 1}, [0, 0, 0])
        )
        # 右柱 (2) — 横梁 (3): Planar-Contact（柱顶 Face 2 — 横梁前面 Face 1）
        Edges.append(
            StructureEdge(2, 3, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 1}, [0, 0, 0])
        )
        # 球网 (4) — 横梁 (3): Planar-Contact（球网前面 Face 1 — 横梁后面 Face 0）
        Edges.append(
            StructureEdge(4, 3, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 0}, [0, 0, 0])
        )
        # 左柱 (1) — 右柱 (2): Alignment（两门柱轴线平行，Cuboid Axis 12 为主轴/竖轴）
        Edges.append(
            StructureEdge(1, 2, "Alignment", {"type": 1, "idx": 12}, {"type": 1, "idx": 12}, [0, 0, 0])
        )
        # 球网 (4) — 横梁 (3): Alignment（球网顶面边 ‖ 横梁轴线）
        # Cuboid 顶面边: Axis 0~3；横梁沿 X 的轴线用 Axis 0（顶面一条边）
        Edges.append(
            StructureEdge(4, 3, "Alignment", {"type": 1, "idx": 0}, {"type": 1, "idx": 0}, [0, 0, 0])
        )
        self.Edges = Edges


class StructureMap_Soccer(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        :param sizes: [B, 13]  (ball [1] + post_L [3] + post_R [3] + bar [3] + net [3])
        :param positions: [B, 15]
        :param rotations: [B, 30]
        Node: 5。Edges: 左柱–横梁、右柱–横梁、球网–横梁 Planar-Contact；左柱–右柱、球网–横梁 Alignment；其余 Free。
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = [Soccer_Mechanism(sizes, positions, rotations)]

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

    print(f"Initializing Soccer (5 parts: ball, posts, bar, net) on {device}...")

    # 球 r~0.026 (geom size .026)
    r_ball = 0.026
    # 门柱：Cuboid(height, top_length, top_width)->(Y,X,Z)。要让柱竖直沿 Z，取 top_width=柱高
    post_xy, post_z = 0.03, 0.12
    # 横梁 (Y=0.03, X=0.2, Z=0.03)：沿 X 跨 0.2，厚度 0.03
    h_bar, l_bar, w_bar = 0.03, 0.2, 0.03
    # 球网：薄长方体板，直接贴在横梁后面
    net_h, net_l, net_w = 0.08, 0.16, 0.005

    sizes_list = [r_ball] + [post_xy, post_xy, post_z] * 2 + [h_bar, l_bar, w_bar] + [net_h, net_l, net_w]
    sizes = torch.tensor([sizes_list], dtype=torch.float32, device=device)

    identity_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    rotations = torch.tensor([identity_6d * 5], dtype=torch.float32, device=device)

    # 球门：左柱、右柱在 X 方向左右分立，横梁架在柱顶；球网贴在横梁后面 (Y 负)
    y_goal = 0.9
    pos_ball = [0.0, 0.6, 0.03]
    half_span_x = (l_bar - post_xy) * 0.5
    pos_post_L = [-half_span_x, y_goal, post_z * 0.5]
    pos_post_R = [half_span_x, y_goal, post_z * 0.5]
    pos_bar = [0.0, y_goal, post_z + w_bar * 0.5]
    # 球网贴在横梁后面，中心与横梁对齐
    pos_net = [0.0, y_goal - w_bar * 0.5 - net_w * 0.5, post_z + w_bar * 0.5]

    positions = torch.tensor(
        [pos_ball + pos_post_L + pos_post_R + pos_bar + pos_net],
        dtype=torch.float32,
        device=device,
    )

    from utils import CLIPEncoder

    try:
        clip_encoder = CLIPEncoder("ViT-B/32").to(device)
        graph = StructureMap_Soccer(sizes, positions, rotations, clip_encoder)

        graph.Node[0].visual_color = [0.95, 0.95, 0.95]
        graph.Node[1].visual_color = [0.2, 0.4, 0.8]
        graph.Node[2].visual_color = [0.2, 0.4, 0.8]
        graph.Node[3].visual_color = [0.2, 0.4, 0.8]
        graph.Node[4].visual_color = [0.9, 0.9, 0.85]

        print(f"Graph: {len(graph.Node)} Nodes, {len(graph.Edge)} Edges")
        from visualization_helper import visualize_structure_graph

        visualize_structure_graph(graph)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
