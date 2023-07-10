import torch
from torch.utils.data import Dataset
from utils import Normalise, RandomCrop, ToTensor, RandomMirror
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class HydranetDataset(Dataset):

    def __init__(self, data_file, transform=None):
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        self.datalist = [x.decode("utf-8").strip("\n").split("\t") for x in datalist]
        self.root_dir = "nyud"
        self.transform = transform
        self.masks_names = ("segm", "depth")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        abs_paths = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]] # Will output list of nyud/*/00000.png
        sample = {}
        sample["image"] = #TODO: Copy/Paste your loaded code

        for mask_name, mask_path in zip(self.masks_names, abs_paths[1:]):
            #TODO: Copy/Paste your loaded code

        if self.transform:
            sample["names"] = self.masks_names
            sample = self.transform(sample)
            # the names key can be removed by the transformation
            if "names" in sample:
                del sample["names"]
        return sample