import torch
import torchvision
# import torchvision.ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
import numpy.random as npr
import os


class FontLoader(DataLoader):
    def __init__(self, path_train: str, path_test: str) -> None:
        super().__init__()
        self.path_train = path_train
        self.path_test = path_test
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=path_train,
            transform=torchvision.transforms.ToTensor()
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=64,
            num_workers=1,
            shuffle=True
        )


a = FontLoader("../converted/train", "../converted/test")
