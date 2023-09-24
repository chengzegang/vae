from torch.utils.data import IterableDataset
import webdataset as wds
import glob
from PIL import Image
from torch import Tensor
import torchvision.transforms.v2.functional as TF
import torch
import os


class Laion5B(IterableDataset):
    def __init__(self, root: str, image_size: int = 256, **kwargs):
        self.root = root
        self.image_size = image_size
        self.data = (
            wds.WebDataset(
                glob.glob(os.path.join(root, "*.tar")),
                wds.ignore_and_continue,
                shardshuffle=True,
            )
            .shuffle(1000)
            .decode("pil")
            .map_dict(jpg=lambda x: self.load_image(x))
        )

    @classmethod
    def from_meta(cls, meta: dict) -> "Laion5B":
        return cls(
            root=meta["root"],
            image_size=meta.get("image_size", 256),
        )

    def load_image(self, path: str) -> Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        image = TF.to_image(image)
        image = TF.to_dtype(image, torch.float32, True)
        return image

    def __len__(self) -> int:
        return 200000000

    def __iter__(self):
        for data in self.data:
            if "jpg" in data:
                jpg = data["jpg"]

                yield jpg
