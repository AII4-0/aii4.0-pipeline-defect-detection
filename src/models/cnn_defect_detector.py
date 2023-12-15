from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics import classification

from src.losses.focal_loss import FocalLoss


class CNNDefectDetector(pl.LightningModule):
    """
    CNN model for defect detection with PyTorch Lightning wrapper
    You can find more information about models here:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html?highlight=lightningmodule#lightningmodule

    Args:
        img_shape (Tuple): Shape of the images
        convs (List[int]): List of convolutional layers
        hiddens (List[int]): List of hidden layers
        dropout (float): Dropout rate
        lr (float): Learning rate
        num_classes (int): Number of classes
    """

    def __init__(
        self,
        img_shape: Tuple,
        convs: List[int],
        hiddens: List[int],
        dropout: float,
        lr: float,
        num_classes: int = 1,
    ):
        super(CNNDefectDetector, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.conv_layers = nn.Sequential()
        for i, (in_channels, out_channels) in enumerate(
            zip([img_shape[0]] + convs[:-1], convs)
        ):
            self.conv_layers.add_module(
                f"conv{i}",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False,
                ),
            )
            self.conv_layers.add_module(f"batchnorm{i}", nn.BatchNorm2d(out_channels))
            self.conv_layers.add_module(f"relu{i}", nn.ReLU(inplace=True))
            self.conv_layers.add_module(
                f"maxpool{i}", nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )

        self.hidden_layers = nn.Sequential()

        last_layer_features = (
            convs[-1]
            * (img_shape[1] // (2 ** len(convs)))
            * (img_shape[2] // (2 ** len(convs)))
        )
        for i, features in enumerate(hiddens):
            self.hidden_layers.add_module(
                f"hidden{i}", nn.Linear(last_layer_features, features)
            )
            self.hidden_layers.add_module(f"relu{i}", nn.ReLU(inplace=True))
            self.hidden_layers.add_module(f"dropout{i}", nn.Dropout(dropout))
            last_layer_features = features

        self.flatten = nn.Flatten()
        self.hidden_layers.add_module(
            "output", nn.Linear(last_layer_features, num_classes)
        )

        self.sigmoid = nn.Sigmoid()
        self.focal_loss = FocalLoss()
        self.train_acc = classification.BinaryAccuracy()
        self.val_acc = classification.BinaryAccuracy()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.hidden_layers(x)
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
