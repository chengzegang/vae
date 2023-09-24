from .gaussian import Gaussian
from .imagefolder import ImageFolder
from .laion5b import Laion5B
from enum import Enum


class Datasets(Enum):
    laion5b = Laion5B
    imagefolder = ImageFolder
