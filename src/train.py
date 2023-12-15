from pathlib import Path

import dvclive
import pytorch_lightning as pl
import torch
import yaml
from torchsummary import summary

from src.data_modules.defect_data_module import DefectDataModule
from src.models import model_registry
from src.utils.callbacks import GradCAMPreviewCallback, PreviewCallback
from src.utils.seed import set_seed


def train() -> None:
    """Train the model"""
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    train_params = params["train"]
    set_seed(train_params["seed"])

    data_module = DefectDataModule(
        dataset_path=Path(prepare_params["dataset_path"]),
        max_images=prepare_params["max_images"],
        imbalance_ratio=prepare_params["imbalance_ratio"],
        img_shape=prepare_params["img_shape"],
        apply_transforms=prepare_params["apply_transforms"],
        train_split=prepare_params["train_split"],
        batch_size=prepare_params["batch_size"],
        seed=train_params["seed"],
        num_workers=train_params["num_workers"],
    )

    model = model_registry[train_params["model_name"]]
    print(model)
    summary(
        model.cuda() if torch.cuda.is_available() else model,
        input_size=tuple(prepare_params["img_shape"]),
        batch_size=prepare_params["batch_size"],
    )
    trainer = pl.Trainer(
        max_epochs=train_params["max_epochs"],
        log_every_n_steps=1,
        callbacks=[
            PreviewCallback(max_images=prepare_params["batch_size"]),
            GradCAMPreviewCallback(max_images=prepare_params["batch_size"]),
        ],
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    # Save model
    trainer.save_checkpoint(train_params["model_save_path"])

    with dvclive.Live(train_params["out_path"]) as live:
        live.log_metric("train_loss", trainer.logged_metrics["train_loss"].item())
        live.log_metric("val_loss", trainer.logged_metrics["val_loss"].item())
        live.log_metric("train_acc", trainer.logged_metrics["train_acc"].item())
        live.log_metric("val_acc", trainer.logged_metrics["val_acc"].item())
        live.log_artifact(train_params["model_save_path"], type="model")
