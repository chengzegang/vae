import glob
import os
from typing import Callable, Iterable

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    def __init__(self, root: str, image_size: int, transform: Callable | None = None):
        self.root = root
        self.image_size = image_size
        self.transform = transform
        self.paths = sorted(self.find(root))

    def find(self, root: str) -> Iterable[str]:
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(("jpg", "jpeg", "png")):
                    yield os.path.join(dirpath, filename)

    @classmethod
    def from_meta(cls, meta: dict) -> "ImageFolder":
        return cls(
            root=meta["root"],
            image_size=meta.get("image_size", 256),
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index) -> Tensor:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        image = TF.to_image(image)
        image = TF.to_dtype(image, torch.float32, True)
        if self.transform is not None:
            image = self.transform(image)

        return image
