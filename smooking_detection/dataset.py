import os
from typing import Optional, List, Tuple, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.vit_model import ViTPreprocessor


class BinarySmokingDataset(Dataset):
    def __init__(self, root: str, transform: Optional[transforms.Compose] = None,
                 class_names: Optional[List[str]] = None):

        self.root = root
        self.transform = transform

        self.class_names = class_names if class_names else ["notsmoking", "smoking"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        self.images: List[str] = []
        self.labels: List[int] = []

        self._load_data()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor

    def _load_data(self):
        for root_dir, dirs, files in os.walk(self.root):
            for file in files:
                if self._is_image_file(file):
                    img_path = os.path.join(root_dir, file)

                    label = self._get_label_from_filename(file)
                    if label is not None:
                        self.images.append(img_path)
                        self.labels.append(label)

    def _is_image_file(self, filename: str) -> bool:
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        return filename.lower().endswith(valid_extensions)

    def _get_label_from_filename(self, filename: str) -> Optional[int]:
        filename_lower = filename.lower()

        if 'notsmoking' in filename_lower or 'not_' in filename_lower or 'no_' in filename_lower:
            return self.class_to_idx.get("notsmoking", 0)
        elif 'smoking' in filename_lower or 'smoke' in filename_lower:
            return self.class_to_idx.get("smoking", 1)

        parts = filename_lower.split('_', 1)
        if len(parts) > 0:
            first_part = parts[0]
            if first_part == "notsmoking" or first_part.startswith("not") or first_part.startswith("no"):
                return self.class_to_idx.get("notsmoking", 0)
            elif first_part == "smoking" or first_part.startswith("smok"):
                return self.class_to_idx.get("smoking", 1)

        return None


class BinarySmokingDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 32,
            num_workers: int = 4,
            image_size: int = 224,
            class_names: Optional[List[str]] = None,
            augment: bool = True,
    ):

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.augment = augment

        self.mean, self.std = ViTPreprocessor.image_mean, ViTPreprocessor.image_std

        self.class_names = class_names if class_names else ["notsmoking", "smoking"]
        self.idx_to_class = {0: 'not_smoking', 1: 'smoking'}
        self.num_classes = len(self.class_names)

        self.train_transform, self.val_transform = self._create_transforms()

        self.train_dataset: Optional[BinarySmokingDataset] = None
        self.val_dataset: Optional[BinarySmokingDataset] = None
        self.test_dataset: Optional[BinarySmokingDataset] = None

        self.class_distribution: Dict[str, Dict[str, int]] = {}

    def get_class_to_name(self):
        return self.idx_to_class

    def _create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        base_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

        if self.augment:
            train_transforms_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        else:
            train_transforms_list = base_transforms.copy()

        val_transforms_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

        train_transforms = transforms.Compose(train_transforms_list)
        val_transforms = transforms.Compose(val_transforms_list)

        return train_transforms, val_transforms

    def prepare_data(self):
        for split in ['Training', 'Validation', 'Testing']:
            split_dir = os.path.join(self.data_dir, split)
            if not os.path.exists(split_dir):
                print(f"Warning: Directory {split_dir} does not exist")
                os.makedirs(split_dir, exist_ok=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_dir = os.path.join(self.data_dir, 'Training')
            val_dir = os.path.join(self.data_dir, 'Validation')

            self.train_dataset = BinarySmokingDataset(
                root=train_dir,
                transform=self.train_transform,
                class_names=self.class_names
            )

            self.val_dataset = BinarySmokingDataset(
                root=val_dir,
                transform=self.val_transform,
                class_names=self.class_names
            )

        if stage == "test" or stage is None:
            test_dir = os.path.join(self.data_dir, 'Testing')

            self.test_dataset = BinarySmokingDataset(
                root=test_dir,
                transform=self.val_transform,
                class_names=self.class_names
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )