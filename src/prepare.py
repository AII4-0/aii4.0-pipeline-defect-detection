import os
from pathlib import Path

import yaml
from dvclive import Live
from PIL import Image

from src.data_modules.defect_data_module import DefectDataModule
from src.utils.seed import set_seed


def prepare() -> None:
    """Prepare the dataset."""
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]
    set_seed(prepare_params["seed"])

    data_module = DefectDataModule(
        dataset_path=Path(prepare_params["dataset_path"]),
        max_images=prepare_params["max_images"],
        imbalance_ratio=prepare_params["imbalance_ratio"],
        img_shape=prepare_params["img_shape"],
        apply_transforms=prepare_params["apply_transforms"],
        train_split=prepare_params["train_split"],
        batch_size=prepare_params["batch_size"],
        seed=prepare_params["seed"],
    )

    data_module.setup(stage=None)
    train_loader = data_module.train_dataloader()

    if not os.path.exists("out/prepared"):
        os.makedirs("out/prepared")

    # Save the first 10 images
    batch = next(iter(train_loader))
    with Live(prepare_params["out_path"]) as live:
        for i, (img, label) in enumerate(zip(batch[0], batch[1])):
            img_arr = (
                (img.permute(1, 2, 0) * 255)
                .cpu()
                .detach()
                .numpy()
                .astype("uint8")
                .squeeze()
            )
            img = Image.fromarray(
                img_arr, mode="RGB" if prepare_params["img_shape"][0] == 3 else "L"
            )
            label_human = "defect" if label == 1 else "normal"
            fp = os.path.join(prepare_params["out_path"], f"img_{i}_{label_human}.png")
            live.log_image(fp, img)
            if i >= 15:
                break
