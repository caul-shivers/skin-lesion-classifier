from typing import List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as models


@dataclass
class FCLayerConfig:
    out_features: int
    activation: Optional[str] = None
    batch_norm: bool = False


class GoogleNetTransferModel(nn.Module):
    def __init__(self, fc_layers: List[FCLayerConfig], num_classes: int = 2):
        super().__init__()
        googlenet = models.googlenet(
            weights=None, aux_logits=False, transform_input=True, init_weights=False
        )
        for p in googlenet.parameters():
            p.requires_grad = False
        self.backbone = googlenet
        self.num_classes = num_classes
        in_features = self.backbone.fc.in_features # 1024
        self.backbone.fc = self._build_fc_layers(fc_layers, in_features)

    @staticmethod
    def get_activation(name):
        return {
            "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "prelu": nn.PReLU(),
            "leakyrelu": nn.LeakyReLU(), "silu": nn.SiLU(), "none": nn.Identity(),
        }[name]

    def _build_fc_layers(self, fc_layers, in_features):
        layers = []
        for layer in fc_layers:
            layers.append(nn.Linear(in_features, layer.out_features))
            if layer.activation is not None:
                layers.append(self.get_activation(layer.activation))
            if layer.batch_norm:
                layers.append(nn.BatchNorm1d(layer.out_features))
            in_features = layer.out_features
        layers.append(nn.Linear(in_features, self.num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)

FINAL_FC_LAYERS = [
    FCLayerConfig(out_features=64, activation='relu', batch_norm=True),
    FCLayerConfig(out_features=32, activation='relu', batch_norm=True),
    FCLayerConfig(out_features=8, activation=None)
]
