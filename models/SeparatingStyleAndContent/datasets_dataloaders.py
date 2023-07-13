import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import numpy.random as npr
import os

from torchvision.io import read_image


class GwFontsDataset(Dataset):
    def __init__(
            self,
            train: bool,
            nsamples: int,
            dataset_len: int,
            converted_dir="../../datasets/gwfonts/converted/"
    ):
        self.directory = converted_dir + ("train" if train else "test")
        self.font_names = [dirname for dirname in os.listdir(self.directory)]
        self.font_len = len(self.font_names)
        self.train = train
        self.test = not self.train
        self.nsamples = nsamples
        self.dataset_len = dataset_len
        self.alphabet_len = 2 * 26 + 10
        if self.test:
            self.train_directory = converted_dir + "train"
            self.train_names = [dirname for dirname in os.listdir(self.train_directory)][:self.nsamples]

        self.content_coord = \
            npr.choice(a=self.alphabet_len, size=self.dataset_len)

        self.style_coord = \
            npr.choice(a=self.font_len, size=self.dataset_len, replace=False)

        self.style_coord_samples = np.zeros(shape=(self.dataset_len, self.nsamples), dtype=int)
        for i in range(self.dataset_len):
            self.style_coord_samples[i] = npr.choice(a=self.alphabet_len - 1, size=self.nsamples, replace=False)
            self.style_coord_samples[i] += self.style_coord_samples[i] >= self.content_coord[i]

        if self.train:
            self.content_coord_samples = np.zeros(shape=(self.dataset_len, self.nsamples), dtype=int)
            for i in range(self.dataset_len):
                self.content_coord_samples[i] = npr.choice(a=self.font_len - 1, size=self.nsamples, replace=False)
                self.content_coord_samples[i] += self.content_coord_samples[i] >= self.style_coord[i]

    def __len__(self):
        return self.dataset_len

    @staticmethod
    def num_to_ord(idx):
        if idx < 10:
            return ord('0') + idx
        if idx < 36:
            return ord('A') + idx - 10
        return ord('a') + idx - 36

    def pict_file(self, style_idx: int, content_idx: int) -> str:
        ord_name = self.num_to_ord(content_idx)
        font_name = self.font_names[style_idx]
        return self.directory + f"/{font_name}/{font_name}_{ord_name}.png"

    def __getitem__(self, idx):
        content_idx = self.content_coord[idx]
        style_idx = self.style_coord[idx]

        pict = read_image(self.pict_file(style_idx, content_idx))

        style_samples = torch.stack([
            read_image(self.pict_file(style_idx, self.style_coord_samples[idx][r]))
            for r in range(self.nsamples)
        ])

        if self.train:
            content_samples = torch.stack([
                read_image(self.pict_file(self.content_coord_samples[idx][r], content_idx))
                for r in range(self.nsamples)
            ])
        else:
            content_samples = torch.stack([
                read_image(self.train_directory +
                           f"/{self.train_names[r]}/{self.train_names[r]}_{self.num_to_ord(content_idx)}.png")
                for r in range(self.nsamples)
            ])

        return (content_samples, style_samples), pict


def create_gw_data(nsamples=10):

    return GwFontsDataset(train=True, nsamples=nsamples, dataset_len=800), \
        GwFontsDataset(train=False, nsamples=nsamples, dataset_len=200)


def create_gw_loaders(nsamples=10, batch_size=4):
    data = create_gw_data(nsamples)

    return DataLoader(data[0], batch_size=batch_size, shuffle=True), \
        DataLoader(data[1], batch_size=batch_size, shuffle=False)