import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image

import os
import random
import numpy as np
from torch.utils.data import Dataset
class TumorDataset(Dataset):
    """ Returns a TumorDataset class object which represents our tumor dataset.
        TumorDataset inherits from torch.utils.data.Dataset class.
    """
    in_channels = 3
    out_channels = 1

    def __init__(self, images_dir, transform=None, image_size=256, subset='train', random_sampling=True, seed=0):
        assert subset in ['all', 'train', 'validation']

        volumes = dict()
        masks = dict()
        print('Loading {} images for analysis...'.format(subset))