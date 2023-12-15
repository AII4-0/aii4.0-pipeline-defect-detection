from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data_modules.defect_data_module import DefectDataModule
from src.datasets.defect_dataset import DefectDataset
from src.utils.seed import seed_worker


class KFoldDefectDataModule(DefectDataModule):
    """
    Data module with KFold defect detection with PyTorch Lightning wrapper

    Args:
        dataset_path (Path): Path to the dataset
        max_images (int): Maximum number of images to use
        img_shape (Tuple): Shape of the images
        apply_transforms (bool): Whether to apply transforms to the images
        train_split (float): Percentage of the dataset to use for training
        batch_size (int): Batch size
    """

    def __init__(
        self,
        k: int,
        num_folds: int,
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
        super(KFoldDefectDataModule, self).__init__(
            dataset_path,
            max_images,
            imbalance_ratio,
            img_shape,
            apply_transforms,
            train_split,
            batch_size,
            seed,
            num_workers,
        )
        self.k = k
        self.num_folds = num_folds

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
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        all_splits = [k for k in kf.split(img_paths)]
        train_indexes, val_indexes = all_splits[self.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.train_dataset = DefectDataset(
            img_paths=[img_paths[i] for i in train_indexes],
            img_shape=self.img_shape,
            apply_transforms=self.apply_transforms,
            max_images=self.max_images,
        )
        self.val_dataset = DefectDataset(
            img_paths=[img_paths[i] for i in val_indexes],
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
