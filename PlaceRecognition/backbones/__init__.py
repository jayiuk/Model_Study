import importlib
import torch.nn as nn

def get_backbone(backbone, **kwargs):
    backbone = getattr(importlib.import_module('torchvision.models'), backbone)(weights='DEFAULT').features
    return nn.Sequential(*list(backbone.children())[:-1])