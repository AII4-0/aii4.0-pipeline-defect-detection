from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.datasets.defect_dataset import DefectDataset
from src.utils.seed import seed_worker


class DefectDataModule(pl.LightningDataModule):
    """
    Data module for defect detection with PyTorch Lightning wrapper
    You can find more information about data modules here:
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html?highlight=datamodule

    Args:
        dataset_path (Path): Path to the dataset
        max_images (int): Maximum number of images to use
        imbalance_ratio (float): Ratio of defect to normal images
        img_shape (Tuple): Shape of the images
        apply_transforms (bool): Whether to apply transforms to the images
        train_split (float): Percentage of the dataset to use for training
        batch_size (int): Batch size
    """

    def __init__(
        self,
        dataset_path: Path,
        max_images: int,
        imbalance_ratio: float,
        img_shape: Tuple,
        apply_transforms: bool,
        train_split: float,
        batch_size: int,
        seed: int,
        num_workers: Optional[int] = None,
    ):
        super(DefectDataModule, self).__init__()
        self.dataset_path = dataset_path
        self.max_images = max_images
        self.imbalance_ratio = imbalance_ratio
        self.img_shape = img_shape
        self.apply_transforms = apply_transforms
        self.train_split = train_split
        self.batch_size = batch_size
        self.seed = seed
        if num_workers is None or num_workers <= 1:
            self.train_num_workers = 1
            self.val_num_workers = 1
        else:
            self.train_num_workers = round(num_workers * 2 / 3)
            self.val_num_workers = round(num_workers * 1 / 3)
        self.gen = torch.Generator().manual_seed(self.seed)

    def setup(self, stage: str):
        img_paths = [
            p
            for p in self.dataset_path.glob("**/*")
            if p.suffix[1:] in ["jpeg", "jpg", "png"]
        ]
        # Normalize the class distribution by resampling the class with fewer images
        normal_paths = [p for p in img_paths if "normal" in p.parent.name]
        defect_paths = [p for p in img_paths if "defect" in p.parent.name]
        if len(normal_paths) > len(defect_paths):
            trim_len = round(len(defect_paths) * self.imbalance_ratio)
            img_paths = normal_paths[:trim_len] + defect_paths
        else:
            trim_len = round(len(normal_paths) * self.imbalance_ratio)
            img_paths = normal_paths + defect_paths[:trim_len]
        print(f"[INFO] Sampled a total of {len(img_paths)} images")
        # Split the dataset
        train_img_paths, val_img_paths = torch.utils.data.random_split(
            img_paths,
            [
                round(len(img_paths) * self.train_split),
                round(len(img_paths) * (1 - self.train_split)),
            ],
        )

        self.train_dataset = DefectDataset(
            img_paths=train_img_paths,
            img_shape=self.img_shape,
            apply_transforms=self.apply_transforms,
            max_images=self.max_images,
        )
        self.val_dataset = DefectDataset(
            img_paths=val_img_paths,
            img_shape=self.img_shape,
            apply_transforms=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=self.gen,
            num_workers=self.train_num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=self.gen,
            num_workers=self.val_num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
