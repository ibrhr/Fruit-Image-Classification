import os
import random
import zipfile
import requests
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FruitsDataset(Dataset):
    """
    PyTorch Dataset for the Fruits-360 100×100 dataset with train/val/test splits.

    Downloads, extracts, and splits the '100x100' Fruits-360 dataset. Automatically creates
    a validation split (10% of training) after download.

    Args:
        root:       Root directory for dataset storage.
        split:      One of {'train', 'val', 'test'}.
        transform:  Optional torchvision transforms to apply to images.
        download:   If True, download & extract if not already present.
    """
    URL = 'https://github.com/fruits-360/fruits-360-100x100/archive/refs/heads/main.zip'
    EXTRACTED_DIR = 'fruits-360-100x100-main'
    SPLIT_DIRS = {'train': 'Training', 'test': 'Test', 'val': 'Validation'}

    def __init__(
        self,
        transform: Optional[Callable] = None,
        root: str = 'dataset',
        split: str = 'train',
        download: bool = False
    ):
        assert split in self.SPLIT_DIRS, f"split must be one of {list(self.SPLIT_DIRS)}"
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if download:
            self._download_and_prepare()

        self.data_dir = self.root / self.EXTRACTED_DIR / self.SPLIT_DIRS[split]
        self.classes, self.class_to_idx = self._find_classes(self.data_dir)
        self.samples = self._make_dataset(self.data_dir, self.class_to_idx)

    def _download_and_prepare(self):
        # Download zip
        self.root.mkdir(parents=True, exist_ok=True)
        r = requests.get(self.URL)
        r.raise_for_status()
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            z.extractall(self.root)

        base = self.root / self.EXTRACTED_DIR
        train_dir = base / self.SPLIT_DIRS['train']
        val_dir = base / self.SPLIT_DIRS['val']
        # Create Validation folder and split 10% of training
        if not val_dir.exists():
            val_dir.mkdir()
            for class_dir in train_dir.iterdir():
                if class_dir.is_dir():
                    
                    images = list(class_dir.glob('*.jpg'))
                    images.sort()                            # ensure deterministic order
                    random.Random(42).shuffle(images)        # reproducible shuffle
                    split_count = max(1, int(0.1 * len(images)))
                    val_images = images[:split_count]
                    
                    # Create class subfolder in val
                    target_class_dir = val_dir / class_dir.name
                    target_class_dir.mkdir(parents=True, exist_ok=True)
                    for img_path in val_images:
                        dest = target_class_dir / img_path.name
                        img_path.rename(dest)

    def _find_classes(self, directory: Path) -> Tuple[List[str], dict]:
        classes = [d.name for d in directory.iterdir() if d.is_dir()]
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, directory: Path, class_to_idx: dict) -> List[Tuple[Path, int]]:
        instances = []
        for class_name, idx in class_to_idx.items():
            for img_path in (directory / class_name).glob('*.jpg'):
                instances.append((img_path, idx))
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
