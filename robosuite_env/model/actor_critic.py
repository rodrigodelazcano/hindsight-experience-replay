from turtle import forward
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingFC(nn.Module):
    
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 use_bias: bool = False,
                 bias_init: float  = 0.0,
                 permutation_invariant: bool = False):
        super(EmbeddingFC, self).__init__()

        layers = []
        layers.append(nn.Linear(in_size, out_size, bias=use_bias))
        layers.append(nn.LayerNorm(out_size))
        
        self._model = nn.Sequential(*layers)
    
    def forward(self, x):
 
        return self._model(x)

class CNNplusFC(nn.Module):
    def __init__(self,
                in_channels: int):
        super(CNNplusFC, self).__init__()
        layers = []
        layers.append(nn.Conv2d( in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten(0,-1))
        layers.append(nn.Linear(in_features=16384, out_features=256))
        layers.append(nn.ReLU())

        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class MLP(nn.Module):
    def __init__(self, input_size=1536 ,hidden_layers = [512, 512, 512]):
        super(MLP, self).__init__()
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_layers[0]))

        last_hidden_size = hidden_layers[0]
        for hidden_size in hidden_layers[1:]:
            layers.append(nn.Linear(last_hidden_size, hidden_size))
            last_hidden_size = hidden_size

        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class Actor(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Actor, self).__init__()
        
        # == RGB INPUT ==
        self.cnn_rgb = CNNplusFC(in_channels=3)
        
        # == DEPTH INPUT ==
        self.conv1_depth = CNNplusFC(in_channels=1)

        # == SEGMENTATION INPUT ==
        self.conv1_seg = CNNplusFC(in_channels=1)
        
        # == ROBOT JOINT STATE INPUT ==
        self.emb_rob_joint = EmbeddingFC(in_size=observation_space["robot_joint_state"].shape[0], out_size=256)

        # == GRIPPER STATE ==
        self.emb_grip = EmbeddingFC(in_size=observation_space["gripper_state"].shape[0], out_size=256)

        # == OBJECT STATE == 
        self.emb_obj = EmbeddingFC(in_size=observation_space["rgb_image"].shape[0], out_size=256)

        self.mlp = MLP()

        self.logits = nn.Linear(512, action_space.shape[0])


    def forward(self, input_dict):
        output_rgb = self.cnn_rgb(input_dict["rgb_image"]/255)
        output_depth = self.cnn_depth(input_dict["depth_image"])
        output_seg = self.cnn_seg(input_dict["segmentation_image"]/torch.max(input_dict["segmentation_image"]))
        output_robot_joint = self.emb_rob_joint(input_dict["robot_joint_state"])
        output_gripper = self.emb_grip(input_dict["gripper_state"])
        output_object = self.emb_obj(input_dict["object_state"])

        x = torch.cat((output_rgb, output_depth, output_seg, output_robot_joint, output_gripper, output_object), dim=1)

        x = self.mlp(x)

        return torch.tanh(self.logits(x))


class Critic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Critic, self).__init__()

        # == RGB INPUT ==
        self.cnn_rgb = CNNplusFC(in_channels=3)
        
        # == DEPTH INPUT ==
        self.conv1_depth = CNNplusFC(in_channels=1)

        # == SEGMENTATION INPUT ==
        self.conv1_seg = CNNplusFC(in_channels=1)
        
        # == ROBOT JOINT STATE INPUT ==
        self.emb_rob_joint = EmbeddingFC(in_size=observation_space["robot_joint_state"].shape[0], out_size=256)

        # == GRIPPER STATE ==
        self.emb_grip = EmbeddingFC(in_size=observation_space["gripper_state"].shape[0], out_size=256)

        # == OBJECT STATE == 
        self.emb_obj = EmbeddingFC(in_size=observation_space["rgb_image"].shape[0], out_size=256)

        self.mlp = MLP(input_size=1536+action_space.shape[0])

        self.q_logits = nn.Linear(512, 1)

    def forward(self, input_dict, action):
        output_rgb = self.cnn_rgb(input_dict["rgb_image"]/255)
        output_depth = self.cnn_depth(input_dict["depth_image"])
        output_seg = self.cnn_seg(input_dict["segmentation_image"]/torch.max(input_dict["segmentation_image"]))
        output_robot_joint = self.emb_rob_joint(input_dict["robot_joint_state"])
        output_gripper = self.emb_grip(input_dict["gripper_state"])
        output_object = self.emb_obj(input_dict["object_state"])

        x = torch.cat((output_rgb, output_depth, output_seg, output_robot_joint, output_gripper, output_object, action), dim=1)

        x = self.mlp(x)

        return self.q_logits(x)