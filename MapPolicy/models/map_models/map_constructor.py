# Construct Structure Map
import torch
import torch.nn as nn
import torch.nn.functional as F

import pathlib
import sys
models_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(
    str(models_path / "maps")
)
from MapPolicy.maps.BinPicking import StructureMap_BinPicking
from MapPolicy.maps.BoxClose import StructureMap_BoxClose
from MapPolicy.maps.CoffeePull import StructureMap_CoffeePull
from MapPolicy.maps.CoffeePush import StructureMap_CoffeePush
from MapPolicy.maps.HandInsert import StructureMap_HandInsert
from MapPolicy.maps.HandlePull import StructureMap_HandlePull
from MapPolicy.maps.HandlePullSide import StructureMap_HandlePullSide
from MapPolicy.maps.PegInsertSide import StructureMap_PegInsertSide
from MapPolicy.maps.PickOutOfHole import StructureMap_PickOutOfHole
from MapPolicy.maps.PickPlace import StructureMap_PickPlace
from MapPolicy.maps.PickPlaceWall import StructureMap_PickPlaceWall
from MapPolicy.maps.Push import StructureMap_Push
from MapPolicy.maps.PushBack import StructureMap_PushBack
from MapPolicy.maps.PushWall import StructureMap_PushWall
from MapPolicy.maps.ReachWall import StructureMap_ReachWall
from MapPolicy.maps.ShelfPlace import StructureMap_ShelfPlace
from MapPolicy.maps.Sweep import StructureMap_Sweep
from MapPolicy.maps.SweepInto import StructureMap_SweepInto
from MapPolicy.models.clip.clip_encoder import CLIPEncoder

MAP_DIM_VOCAB = {
    "bin-picking": [19, 34, 64],
    "box-close": [20, 38, 74],
    "coffee-pull": [11, 23, 47],
    "coffee-push": [11, 23, 47],
    "hand-insert": [6, 12, 24],
    "handle-pull": [6, 15, 33],
    "handle-pull-side": [6, 15, 33],
    "peg-insert-side": [8, 17, 35],
    "pick-out-of-hole": [9, 18, 36],
    "pick-place": [3, 6, 12],
    "pick-place-wall": [9, 18, 36],
    "push": [3, 6, 12],
    "push-back": [3, 6, 12],
    "push-wall": [9, 18, 36],
    "reach-wall": [6, 12, 24],
    "shelf-place": [14, 26, 50],
    "sweep": [3, 6, 12],
    "sweep-into": [9, 18, 36],
}
MAP_CLASS_VOCAB = { 
    "bin-picking": StructureMap_BinPicking,
    "box-close": StructureMap_BoxClose,
    "coffee-pull": StructureMap_CoffeePull,
    "coffee-push": StructureMap_CoffeePush,    
    "hand-insert": StructureMap_HandInsert,
    "handle-pull": StructureMap_HandlePull,
    "handle-pull-side": StructureMap_HandlePullSide,
    "peg-insert-side": StructureMap_PegInsertSide,
    "pick-out-of-hole": StructureMap_PickOutOfHole,
    "pick-place": StructureMap_PickPlace,
    "pick-place-wall": StructureMap_PickPlaceWall,
    "push": StructureMap_Push,
    "push-back": StructureMap_PushBack,
    "push-wall": StructureMap_PushWall,
    "reach-wall": StructureMap_ReachWall,
    "shelf-place": StructureMap_ShelfPlace,
    "sweep": StructureMap_Sweep,
    "sweep-into": StructureMap_SweepInto,
}

class ParameterEstimator_SingleFrame(nn.Module):
    def __init__ (self,
                  point_cloud_encoder: nn.Module,
                  map_name,
                  device,
                  ):
        if map_name not in MAP_DIM_VOCAB or map_name not in MAP_CLASS_VOCAB:
            raise ValueError(f"Unknown map_name: {map_name}. Available: {list(MAP_CLASS_VOCAB.keys())}")
        self.map_name = map_name
        self.dims = MAP_DIM_VOCAB[map_name]
        self.device = device
        self.MapClass = MAP_CLASS_VOCAB[map_name]
        super(ParameterEstimator_SingleFrame, self).__init__()
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(self.device)
        self.point_cloud_encoder = point_cloud_encoder
        self.estimation_head = nn.Sequential(
            nn.Linear(point_cloud_encoder.feature_dim, self.dims[2]),
        )
        
    def forward(self, point_cloud):
        features = self.point_cloud_encoder(point_cloud)
        parameters = self.estimation_head(features)
        sizes = parameters[:, 0:self.dims[0]]
        positions = parameters[:, self.dims[0]:self.dims[1]]
        rotations = parameters[:, self.dims[1]:self.dims[2]]
        structure_map = self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=False)
        return structure_map
    
class ParameterEstimator_SingleFrame_Regularization(nn.Module):
    def __init__ (self,
                  point_cloud_encoder: nn.Module,
                  map_name,
                  device,
                  ):
        if map_name not in MAP_DIM_VOCAB or map_name not in MAP_CLASS_VOCAB:
            raise ValueError(f"Unknown map_name: {map_name}. Available: {list(MAP_CLASS_VOCAB.keys())}")
        self.map_name = map_name
        self.dims = MAP_DIM_VOCAB[map_name]
        self.device = device
        self.MapClass = MAP_CLASS_VOCAB[map_name]
        super().__init__()
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(self.device)
        self.point_cloud_encoder = point_cloud_encoder
        self.estimation_head = nn.Sequential(
            nn.Linear(point_cloud_encoder.feature_dim, self.dims[2]),
        )
        
    def forward(self, point_cloud):
        features = self.point_cloud_encoder(point_cloud)
        parameters = self.estimation_head(features)
        sizes = parameters[:, 0:self.dims[0]]
        positions = parameters[:, self.dims[0]:self.dims[1]]
        rotations = parameters[:, self.dims[1]:self.dims[2]]
        structure_map = self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=True)
        return structure_map
