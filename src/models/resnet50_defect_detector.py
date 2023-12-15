from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics import classification
from torchvision.models import ResNet50_Weights, resnet50

from src.losses.focal_loss import FocalLoss


class ResNet50DefectDetector(pl.LightningModule):
    """
    ResNet50 model for defect detection with PyTorch Lightning wrapper
    You can find more information about models here:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html?highlight=lightningmodule#lightningmodule

    Args:
        img_shape (Tuple): Shape of the images
        hiddens (List[int]): List of hidden layers
        dropout (float): Dropout rate
        lr (float): Learning rate
        num_classes (int): Number of classes
    """

    def __init__(
        self,
        img_shape: Tuple,
        hiddens: List[int],
        dropout: float,
        lr: float,
        num_classes: int = 1,
    ):
        super(ResNet50DefectDetector, self).__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.requires_grad_(False)
        if img_shape[0] == 1:
            self.resnet50.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

        hidden_layers = nn.Sequential()

        last_layer_features = self.resnet50.fc.in_features
        for i, features in enumerate(hiddens):
            hidden_layers.add_module(
                f"hidden{i}", nn.Linear(last_layer_features, features)
            )
            hidden_layers.add_module(f"relu{i}", nn.ReLU(inplace=True))
            hidden_layers.add_module(f"dropout{i}", nn.Dropout(dropout))
            last_layer_features = features

        hidden_layers.add_module("output", nn.Linear(last_layer_features, num_classes))

        self.resnet50.fc = hidden_layers
        self.sigmoid = nn.Sigmoid()
        self.focal_loss = FocalLoss()
        self.train_acc = classification.BinaryAccuracy()
        self.val_acc = classification.BinaryAccuracy()

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet50(x)
        return self.sigmoid(x)

    def training_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.focal_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_hat, y), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.focal_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(y_hat, y), on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
