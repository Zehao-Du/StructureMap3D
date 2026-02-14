import torch

from Structure_Primitive import Cylinder, Torus
from base_template import StructureEdge, StructureGraph


class Peg:
    def __init__(self, sizes, positions, rotations):
        semantic = "peg"

        Nodes = []
        Edges = []

        # sizes: [B, 2] = [peg_height, peg_radius]
        if sizes.shape[1] != 2:
            raise ValueError(f"Expected peg sizes shape [B, 2], got {tuple(sizes.shape)}")
        if positions.shape[1] != 3:
            raise ValueError(f"Expected peg positions shape [B, 3], got {tuple(positions.shape)}")
        if rotations.shape[1] != 6:
            raise ValueError(f"Expected peg rotations shape [B, 6], got {tuple(rotations.shape)}")

        peg_height = sizes[:, 0:1]
        peg_radius = sizes[:, 1:2]
        size_peg = torch.cat([peg_height, peg_radius, peg_radius], dim=1)

        Nodes.append(
            Cylinder(
                height=size_peg[:, 0],
                top_radius=size_peg[:, 1],
                position=positions,
                rotation=rotations,
                Semantic=semantic,
            )
        )

        self.Nodes = Nodes
        self.Edges = Edges


class Nut:
    def __init__(self, sizes, positions, rotations):
        semantic_nut = "nut"
        semantic_handle = "nut handle"

        Nodes = []
        Edges = []

        # sizes: [B, 4] = [nut_central_radius, nut_tube_radius, handle_radius, handle_height]
        if sizes.shape[1] != 4:
            raise ValueError(f"Expected nut sizes shape [B, 4], got {tuple(sizes.shape)}")
        # positions: [B, 6] = [nut_xyz, handle_xyz]
        if positions.shape[1] != 6:
            raise ValueError(f"Expected nut positions shape [B, 6], got {tuple(positions.shape)}")
        # rotations: [B, 12] = [nut_6d, handle_6d]
        if rotations.shape[1] != 12:
            raise ValueError(f"Expected nut rotations shape [B, 12], got {tuple(rotations.shape)}")

        nut_central_radius = sizes[:, 0]
        nut_tube_radius = sizes[:, 1]
        handle_radius = sizes[:, 2]
        handle_height = sizes[:, 3]
        nut_position = positions[:, 0:3]
        handle_pos = positions[:, 3:6]
        nut_rotation = rotations[:, 0:6]
        handle_rotation = rotations[:, 6:12]
        
        # Nut: Torus (donut) around local Y-axis
        Nodes.append(
            Torus(
                central_radius=nut_central_radius,
                start_torus_radius=nut_tube_radius,
                position=nut_position,
                rotation=nut_rotation,
                Semantic=semantic_nut,
            )
        )

        Nodes.append(
            Cylinder(
                height=handle_height,
                top_radius=handle_radius,
                position=handle_pos,
                rotation=handle_rotation,
                Semantic=semantic_handle,
            )
        )

        # Fixed connection between nut (torus) and handle (cylinder)
        # Torus Face idx=0 (Outer), Cylinder Face idx=0 (Top)
        B = sizes.shape[0]
        device = sizes.device
        Edges.append(
            StructureEdge(
                0,
                1,
                "Fixed",
                {"type": 0, "idx": 0},
                {"type": 0, "idx": 0},
                torch.zeros((B, 3), device=device),
            )
        )

        self.Nodes = Nodes
        self.Edges = Edges


class StructureMap_Disassemble(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        Minimal structure map for MetaWorld disassemble.

        sizes: [B, 6]
            [peg_height, peg_radius, nut_central_radius, nut_tube_radius, handle_radius, handle_height]
        positions: [B, 9] (3 nodes * 3)
        rotations: [B, 18] (3 nodes * 6D rotation)
        total: [B, 33]
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        # sizes: [B, 6] = [peg_height, peg_radius, nut_central_radius, nut_tube_radius, handle_radius, handle_height]
        if sizes.shape[1] != 6:
            raise ValueError(f"Expected sizes shape [B, 6], got {tuple(sizes.shape)}")
        if positions.shape[1] != 9:
            raise ValueError(f"Expected positions shape [B, 9], got {tuple(positions.shape)}")
        if rotations.shape[1] != 18:
            raise ValueError(f"Expected rotations shape [B, 18], got {tuple(rotations.shape)}")

        peg_obj = Peg(sizes[:, 0:2], positions[:, 0:3], rotations[:, 0:6])
        nut_obj = Nut(sizes[:, 2:6], positions[:, 3:9], rotations[:, 6:18])
        
        Objects = [peg_obj, nut_obj]

        Nodes = []
        Edges = []
        num_node = 0
        for obj in Objects:
            Nodes.extend(obj.Nodes)
            for edge in obj.Edges:
                edge.update_node_idx(num_node)
                Edges.append(edge)
            num_node += len(obj.Nodes)

        # Cylindrical constraint between peg (node 0) and nut main body (node 1)
        B = sizes.shape[0]
        device = sizes.device
        Edges.append(
            StructureEdge(
                0,
                1,
                "Cylindrical",
                {"type": 1, "idx": 0},
                {"type": 1, "idx": 0},
                torch.zeros((B, 3), device=device),
            )
        )

        super().__init__(Nodes, Edges, clip_model)

    def _preprocess_parameters(self, sizes):
        size_range = (0.02, 5)
        min_s, max_s = size_range
        return torch.sigmoid(sizes) * (max_s - min_s) + min_s
