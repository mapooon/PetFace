import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights


def r50(n_classes=512):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, n_classes),nn.BatchNorm1d(n_classes))
    return model

def r101(n_classes=512):
    weights = ResNet101_Weights.DEFAULT
    model = resnet101(weights=weights)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, n_classes),nn.BatchNorm1d(n_classes))
    return model