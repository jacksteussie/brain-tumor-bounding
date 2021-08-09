from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image

import os
import random


class TumorDataset(Dataset):
    """ Returns a TumorDataset class object which represents our tumor dataset.
    TumorDataset inherits from torch.utils.data.Dataset class.
    """