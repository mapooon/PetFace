import torch
from torch import nn
from torchvision.models import vit_b_32, ViT_B_32_Weights

def vitb():
    weights = ViT_B_32_Weights.DEFAULT
    model = vit_b_32(weights=weights)
    head_bn=nn.BatchNorm1d(512, eps=1e-05)
    nn.init.constant_(head_bn.weight, 1.0)
    head_bn.weight.requires_grad = False
    model.heads = nn.Sequential(nn.Linear(model.hidden_dim, 512),head_bn)
    
    # model.heads = vit_b_32(num_classes=n_classes).heads
    # head=model.heads
    
    return model