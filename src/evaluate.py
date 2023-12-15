from pathlib import Path

import dvclive
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
from torchmetrics import classification

from src.data_modules.defect_data_module import DefectDataModule
from src.models import model_registry
from src.utils.seed import set_seed


def evaluate() -> None:
    """Evaluate the dataset."""
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    train_params = params["train"]
    evaluate_params = params["evaluate"]
    set_seed(train_params["seed"])

    # Load model
    model: pl.LightningModule = model_registry[train_params["model_name"]]
    model = type(model).load_from_checkpoint(train_params["model_save_path"])
    model.eval()

    metrics = {
        "val_acc": classification.BinaryAccuracy(),
        "val_prec": classification.BinaryPrecision(),
        "val_recall": classification.BinaryRecall(),
        "val_f1": classification.BinaryF1Score(),
        "val_auc": classification.BinaryAUROC(),
    }
    confmat = classification.BinaryConfusionMatrix(normalize="all")
    roccurve = classification.BinaryROC()

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
    data_module.setup("validate")
    with torch.no_grad():
        for batch in data_module.val_dataloader():
            x, y = batch
            y_hat = model(x)
            for metric in metrics.values():
                metric(y_hat, y)
            confmat(y_hat, y)
            roccurve(y_hat, y.int())

    with dvclive.Live(evaluate_params["out_path"]) as live:
        for metric_name, metric in metrics.items():
            live.log_metric(metric_name, metric.compute().item())
        # Confusion matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        confmat.plot(ax=ax, labels=["Normal", "Defect"])
        live.log_image("confusion_matrix", fig)
        # ROC curve
        fig, ax = plt.subplots(figsize=(8, 8))
        roccurve.plot(ax=ax, score=True)
        live.log_image("roc_curve", fig)
