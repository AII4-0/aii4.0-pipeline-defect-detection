import torch
from torch import nn
from torch.nn import functional as F


# Source from: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Focal-Loss
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss
