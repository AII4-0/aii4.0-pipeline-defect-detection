#################################################################################
# NOTE: This file is not used in the pipeline but serves as an example of how to
#       use KFold cross validation with PyTorch Lightning.
#################################################################################
from pathlib import Path

import dvclive
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import yaml
from torchsummary import summary

from src.data_modules.kfold_defect_data_module import KFoldDefectDataModule
from src.models import model_registry
from src.utils.seed import set_seed


def train() -> None:
    """Train the model"""
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    train_params = params["train"]
    set_seed(train_params["seed"])

    # Print model summary
    model = model_registry[train_params["model_name"]]
    print(model)
    summary(
        model,
        input_size=tuple(prepare_params["img_shape"]),
        batch_size=prepare_params["batch_size"],
    )

    # KFold cross validation training
    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_prec": [],
        "val_recall": [],
        "val_confmat": [],
    }
    num_folds = train_params["num_folds"]
    for k in range(num_folds):
        data_module = KFoldDefectDataModule(
            k=k,
            num_folds=num_folds,
            dataset_path=Path(prepare_params["dataset_path"]),
            max_images=prepare_params["max_images"],
            img_shape=prepare_params["img_shape"],
            apply_transforms=prepare_params["apply_transforms"],
            train_split=prepare_params["train_split"],
            batch_size=prepare_params["batch_size"],
            seed=train_params["seed"],
            num_workers=train_params["num_workers"],
        )

        model = model_registry[train_params["model_name"]]

        trainer = pl.Trainer(
            max_epochs=train_params["max_epochs"],
            log_every_n_steps=1,
            # callbacks=[PreviewCallback(), GradCAMPreviewCallback()],
        )
        trainer.fit(
            model=model,
            datamodule=data_module,
        )
        results["train_loss"].append(trainer.logged_metrics["train_loss"].numpy())
        results["train_acc"].append(trainer.logged_metrics["train_acc"].numpy())
        results["val_loss"].append(trainer.logged_metrics["val_loss"].numpy())
        results["val_acc"].append(trainer.logged_metrics["val_acc"].numpy())
        results["val_prec"].append(trainer.logged_metrics["val_prec"].numpy())
        results["val_recall"].append(trainer.logged_metrics["val_recall"].numpy())
        results["val_confmat"].append(model.val_confmat.compute().numpy())

    avg_val_confmat = np.mean(results["val_confmat"], axis=0)
    confmat_fig = plt.figure(figsize=(10, 10))
    ax = confmat_fig.add_subplot(1, 1, 1)
    sns.heatmap(avg_val_confmat, annot=True, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    with dvclive.Live("out/train") as live:
        live.log_metric("train_loss", np.mean(results["train_loss"]))
        live.log_metric("val_loss", np.mean(results["val_loss"]))
        live.log_metric("train_acc", np.mean(results["train_acc"]))
        live.log_metric("val_acc", np.mean(results["val_acc"]))
        live.log_metric("val_prec", np.mean(results["val_prec"]))
        live.log_metric("val_recall", np.mean(results["val_recall"]))
        live.log_image("confusion_matrix", confmat_fig)
