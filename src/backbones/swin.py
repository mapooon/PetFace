import torch
from torch import nn
from torchvision.models import swin_b, Swin_B_Weights

def swinb():
    weights = Swin_B_Weights.DEFAULT
    model = swin_b(weights=weights)
    head_bn=nn.BatchNorm1d(512, eps=1e-05)
    nn.init.constant_(head_bn.weight, 1.0)
    head_bn.weight.requires_grad = False
    model.head = nn.Sequential(nn.Linear(model.head.in_features, 512),head_bn)
    
    return model