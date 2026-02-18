import os
import pathlib
import sys

import clip
import torch
import torch.nn as nn
from torchvision.transforms import Normalize


class CLIPEncoder(nn.Module):
    def __init__(self, model_name, freeze=True):
        super(CLIPEncoder, self).__init__()
        self.model, self.preprocess = clip.load(model_name)
        # see: https://github.com/openai/CLIP/blob/main/clip/clip.py line 79
        self.preprocess = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        self.feature_dim = {
            "ViT-B/32": 512,
        }[model_name]
        if freeze:
            self.freeze()

    def forward(self, x):
        # x = self.preprocess(x)
        return self.model.encode_text(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False