from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DefectDataset(Dataset):
    """
    PyTorch dataset for defect detection

    Args:
        dataset_path (Path): Path to the dataset
        img_shape (Tuple): Shape of the images
        apply_transforms (bool): Whether to apply transforms to the images
    """

    def __init__(
        self,
        img_paths: Path,
        img_shape: Tuple,
        apply_transforms: bool,
        max_images: int = None,
    ):
        self.img_paths = img_paths
        self.img_shape = img_shape
        self.apply_transforms = apply_transforms
        self.max_images = max_images

        self.base_transform_fisrt = transforms.Compose(
            [transforms.Resize((img_shape[1], img_shape[2]))]
        )
        self.base_transform_last = transforms.Compose([transforms.ToTensor()])
        self.augment_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (img_shape[1], img_shape[2]), scale=(0.95, 1.0)
                ),
            ]
        )

    def __len__(self):
        if self.max_images is not None:
            return self.max_images
        else:
            return len(self.img_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Loop through the images
        path = self.img_paths[idx % len(self.img_paths)]
        img = Image.open(path)
        label = 1 if path.parent.name == "defect" else 0
        if self.img_shape[0] == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = self.base_transform_fisrt(img)

        if self.apply_transforms:
            img = self.augment_transform(img)

        img = self.base_transform_last(img)
        return img, torch.tensor([label], dtype=torch.float32)
