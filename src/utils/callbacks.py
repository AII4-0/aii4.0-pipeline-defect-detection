from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models.cnn_defect_detector import CNNDefectDetector
from src.utils.image import label_pred_images


class PreviewCallback(pl.Callback):
    """Lightning callback to save a preview of the model's predictions"""

    def __init__(self, max_images: int = 32) -> None:
        super().__init__()
        self.max_images = max_images
        self.batch: Optional[tuple] = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Called when the validation epoch ends"""
        if self.batch is None:
            val_dataloader = trainer.datamodule.val_dataloader()
            train_batch = next(iter(trainer.datamodule.train_dataloader()))
            # Grab a few validation samples from the dataloader
            self.batch = next(iter(val_dataloader))

        x, y = self.batch
        # Forward pass
        y_hat = pl_module(x.to(pl_module.device))

        if trainer.global_step == 0:  # only save input and target images the first time
            trainer.logger.experiment.add_images(
                "train_input",
                train_batch[0][: self.max_images, :3, :, :],
                trainer.global_step,
            )
            trainer.logger.experiment.add_images(
                "val_input",
                x[: self.max_images, :3, :, :],
                trainer.global_step,
            )
        trainer.logger.experiment.add_images(
            "val_pred",
            label_pred_images(x[: self.max_images, :3, :, :], y, y_hat),
            trainer.global_step,
        )


class GradCAMPreviewCallback(pl.Callback):
    """Lightning callback to save a preview of the model activation maps"""

    def __init__(self, max_images: int = 32) -> None:
        super().__init__()
        self.max_images = max_images
        self.batch: Optional[tuple] = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Called when the validation epoch ends"""
        if self.batch is None:
            val_dataloader = trainer.datamodule.val_dataloader()
            # Grab a few validation samples from the dataloader
            self.batch = next(iter(val_dataloader))

        x, _ = self.batch

        # GradCAM
        assert isinstance(
            pl_module, CNNDefectDetector
        ), "Update the target layers for another model"
        target_layers = [
            pl_module.conv_layers.conv0,
            pl_module.conv_layers.conv1,
            pl_module.conv_layers.conv2,
            pl_module.conv_layers.conv3,
        ]
        batch = np.ndarray((x.shape[0], 3, x.shape[-2], x.shape[-1]), dtype=np.float32)
        with GradCAM(model=pl_module, target_layers=target_layers) as cam:
            torch.set_grad_enabled(True)
            grayscale_cam = cam(input_tensor=x.to(pl_module.device))
            for i in range(max(len(x), self.max_images)):
                arr_img = x[i].permute(1, 2, 0).cpu().detach().numpy()
                # convert to rgb
                rgb_img = np.concatenate((arr_img, arr_img, arr_img), axis=-1)
                visualization = show_cam_on_image(
                    rgb_img, grayscale_cam[i], use_rgb=True
                )
                visualization = visualization.transpose(2, 0, 1)
                batch[i] = visualization / 255
        trainer.logger.experiment.add_images(
            "gradcam",
            batch,
            trainer.global_step,
        )
