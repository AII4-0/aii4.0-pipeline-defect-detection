from pathlib import Path

import bentoml
import onnx
import pytorch_lightning as pl
import torch
import yaml

from src.models import model_registry


def export() -> None:
    """Export the model to the ONNX format."""
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    train_params = params["train"]
    export_params = params["export"]

    # Load model
    model: pl.LightningModule = model_registry[train_params["model_name"]]
    model = type(model).load_from_checkpoint(train_params["model_save_path"])
    model.eval()

    input_sample = torch.randn(
        (prepare_params["batch_size"], *prepare_params["img_shape"]),
        dtype=torch.float32,
    )
    export_path = Path(export_params["out_path"]) / "model.onnx"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    model.to_onnx(
        export_path,
        input_sample,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    onnx_model = onnx.load(export_path)
    bentoml.onnx.save_model(
        "onnx_defect_detection_api",
        onnx_model,
        signatures={
            "run": {"batchable": True},
        },
    )
