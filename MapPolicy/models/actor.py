import abc
from typing import List
from functools import partial

# import clip
import numpy as np
import torch
import torch.nn as nn

from MapPolicy.helpers.graphics import PointCloud
from MapPolicy.models.mlp.batchnorm_mlp import BatchNormMLP
from MapPolicy.models.mlp.mlp import MLP
from MapPolicy.models.map_models.GNN_encoder import RoboGraphormerLayer


class Actor(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, images, point_clouds, robot_states):
        pass


class DemoModel(Actor):

    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        map_constructor,
        map_encoder,
        point_cloud_dropout_rate: float,
        robot_state_dim: int,
        robot_state_dropout_rate: float,
        action_dim: int,
        policy_hidden_dims: List[int],
        policy_head_init_method: str,
    ):
        super(DemoModel, self).__init__()
        self.point_cloud_encoder = point_cloud_encoder
        self.map_constructor = map_constructor
        self.map_encoder = map_encoder
        self.point_cloud_dropout = nn.Dropout(point_cloud_dropout_rate)
        self.robot_state_encoder = nn.Linear(
            robot_state_dim, point_cloud_encoder.feature_dim
        )
        self.robot_state_dropout = nn.Dropout(robot_state_dropout_rate)
        self.policy_head = MLP(
            input_dim=2 * point_cloud_encoder.feature_dim + 128,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            init_method=policy_head_init_method,
        )

    def forward(self, images, point_clouds, robot_states, texts):
        # * Notice: normalize the input point cloud
        point_clouds = PointCloud.normalize(point_clouds)
        
        # Visual Encode
        point_cloud_emb = self.point_cloud_encoder(point_clouds)
        point_cloud_emb = self.point_cloud_dropout(point_cloud_emb)
        
        # Map Encode
        structure_map = self.map_constructor(point_clouds)
        map_emb = self.map_encoder(structure_map, point_cloud_emb)
        
        # Robot State Encode
        robot_state_emb = self.robot_state_encoder(robot_states)
        robot_state_emb = self.robot_state_dropout(robot_state_emb)
        
        # Fusion
        emb = torch.cat([point_cloud_emb, map_emb, robot_state_emb], dim=1)
        
        # Policy (TODO: isolate policy)
        actions = self.policy_head(emb)
        
        return actions

class SingleFrame_MLP(Actor):
    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        map_constructor: nn.Module,
        map_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,
        policy_hidden_dims: List[int],
        policy_head_init_method: str,
    ):
        super(SingleFrame_MLP, self).__init__()
        self.point_cloud_encoder = point_cloud_encoder
        self.map_constructor = map_constructor
        self.map_encoder = map_encoder
        self.robot_state_dim = robot_state_dim
        self.robot_state_encoder = nn.Linear(
            robot_state_dim, point_cloud_encoder.feature_dim
        )
        self.policy_head = MLP(
            input_dim=2 * point_cloud_encoder.feature_dim + map_encoder.feature_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            init_method=policy_head_init_method,
        )
    def forward(self, images, point_clouds, point_cloud_no_robot, robot_states, texts):
        # * Notice: normalize the input point cloud
        point_clouds = PointCloud.normalize(point_clouds)
        
        # Visual Encode
        point_cloud_emb = self.point_cloud_encoder(point_clouds)
        
        # Map Encode
        structure_map = self.map_constructor(point_cloud_no_robot)
        map_emb = self.map_encoder(structure_map)
        
        # Robot State Encode
        robot_state_emb = self.robot_state_encoder(robot_states)
        
        # Fusion
        emb = torch.cat([point_cloud_emb, map_emb, robot_state_emb], dim=1)
        
        # Policy (TODO: isolate policy)
        actions = self.policy_head(emb)
        
        return actions
    
class SingleFrame_MLP_Chamferloss(Actor):
    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        map_constructor: nn.Module,
        map_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,
        policy_hidden_dims: List[int],
        policy_head_init_method: str,
        loss_map_construction,
    ):
        super().__init__()
        self.point_cloud_encoder = point_cloud_encoder
        self.map_constructor = map_constructor
        self.map_encoder = map_encoder
        self.robot_state_dim = robot_state_dim
        self.robot_state_encoder = nn.Linear(
            robot_state_dim, point_cloud_encoder.feature_dim
        )
        self.policy_head = MLP(
            input_dim=2 * point_cloud_encoder.feature_dim + map_encoder.feature_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            init_method=policy_head_init_method,
        )
        self.loss_map_construction = loss_map_construction
    def forward(self, images, point_clouds, point_cloud_no_robot, robot_states, texts):
        # * Notice: normalize the input point cloud
        point_clouds = PointCloud.normalize(point_clouds)
        
        # Visual Encode
        point_cloud_emb = self.point_cloud_encoder(point_clouds)
        
        # Map Encode
        structure_map = self.map_constructor(point_cloud_no_robot)
        map_emb = self.map_encoder(structure_map)
        
        # Robot State Encode
        robot_state_emb = self.robot_state_encoder(robot_states)
        
        # Fusion
        emb = torch.cat([point_cloud_emb, map_emb, robot_state_emb], dim=1)
        
        # Policy (TODO: isolate policy)
        actions = self.policy_head(emb)
        
        if self.training:
            # Map Loss
            point_cloud_map = structure_map.complete_point_cloud()
            loss_map = self.loss_map_construction(point_cloud_no_robot, point_cloud_map)
            return actions, loss_map
        
        return actions

class SingleFrame_GNN(Actor):
    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        map_constructor: nn.Module,
        map_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,
        policy_hidden_dims: List[int],
        policy_head_init_method: str,
        loss_map_construction,
    ):
        super().__init__()
        self.point_cloud_encoder = point_cloud_encoder
        self.map_constructor = map_constructor
        self.map_encoder = map_encoder
        self.robot_state_dim = robot_state_dim
        self.robot_state_encoder = nn.Linear(
            robot_state_dim, point_cloud_encoder.feature_dim
        )
        self.policy_head = MLP(
            input_dim=2 * point_cloud_encoder.feature_dim + map_encoder.feature_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            init_method=policy_head_init_method,
        )
        self.loss_map_construction = loss_map_construction
    def forward(self, images, point_clouds, point_cloud_no_robot, robot_states, texts):
        # * Notice: normalize the input point cloud
        point_clouds = PointCloud.normalize(point_clouds)
        point_cloud_no_robot = PointCloud.normalize(point_cloud_no_robot)
        
        # Visual Encode
        point_cloud_emb = self.point_cloud_encoder(point_clouds)
        
        # Map Encode
        structure_map = self.map_constructor(point_cloud_no_robot)
        map_emb, math_loss = self.map_encoder(structure_map.data)
        
        # Robot State Encode
        robot_state_emb = self.robot_state_encoder(robot_states)
        
        # Fusion
        emb = torch.cat([point_cloud_emb, map_emb, robot_state_emb], dim=1)
        
        # Policy (TODO: isolate policy)
        actions = self.policy_head(emb)
        
        if self.training:
            # Map Loss
            point_cloud_map = structure_map.complete_point_cloud()
            loss_map = self.loss_map_construction(point_cloud_no_robot, point_cloud_map)
            # Math Loss
            loss_math = math_loss['math_loss'] + math_loss['ortho_loss']
            return actions, loss_map, loss_math
        
        return actions
    
class SingleFrame_Lift3d_GNN(Actor):
    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        map_constructor,
        map_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,
        policy_hidden_dims: List[int],
        policy_head_init_method: str,
        loss_map_construction,
    ):
        super().__init__()
        self.point_cloud_encoder = point_cloud_encoder
        
        if isinstance(map_constructor, partial) or callable(map_constructor):
            self.map_constructor = map_constructor(point_cloud_encoder=self.point_cloud_encoder)
        else:
            self.map_constructor = map_constructor

        self.map_encoder = map_encoder
        self.robot_state_dim = robot_state_dim
        self.robot_state_encoder = nn.Linear(
            robot_state_dim, point_cloud_encoder.feature_dim
        )
        self.policy_head = MLP(
            input_dim=2 * point_cloud_encoder.feature_dim + map_encoder.feature_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            init_method=policy_head_init_method,
        )
        self.loss_map_construction = loss_map_construction
    def forward(self, images, point_clouds, point_cloud_no_robot, robot_states, texts):
        # * Notice: normalize the input point cloud
        point_clouds = PointCloud.normalize(point_clouds)
        point_cloud_no_robot = PointCloud.normalize(point_cloud_no_robot)
        
        # Visual Encode
        point_cloud_emb = self.point_cloud_encoder(point_clouds)
        
        # Map Encode
        structure_map = self.map_constructor(point_cloud_no_robot)
        map_emb, math_loss = self.map_encoder(structure_map.data)
        
        # Robot State Encode
        robot_state_emb = self.robot_state_encoder(robot_states)
        
        # Fusion
        emb = torch.cat([point_cloud_emb, map_emb, robot_state_emb], dim=1)
        
        # Policy (TODO: isolate policy)
        actions = self.policy_head(emb)
        
        if self.training:
            # Map Loss
            point_cloud_map = structure_map.complete_point_cloud()
            loss_map = self.loss_map_construction(point_cloud_no_robot, point_cloud_map)
            # Math Loss
            loss_math = math_loss['math_loss'] + math_loss['ortho_loss']
            return actions, loss_map, loss_math
        
        return actions