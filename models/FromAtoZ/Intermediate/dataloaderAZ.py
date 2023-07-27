from random import triangular
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import numpy.random as npr
import os

import pandas as pd
import cv2

import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image, ImageChops, ImageDraw, ImageOps
from torchvision.io import read_image

pic_sz = 80

'''
    one font is one batch. After that it can be sliced into subsets.  
'''

def ping():
    return "pong"

def get_ord(name):
    return name.slice("_")[-2]

def convert(r_image):
    numpy_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(numpy_image, (pic_sz, pic_sz)).astype(np.float32)
    image = image.reshape(-1)
    return image

def getImage(font:str, char: str):
    return sImage.open(f"gwfonts/converted/train/{font}/{font}_{ord(char)}.png")


def resize(image):
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if width != height:
        bigside = width if width > height else height

        background = Image.new('RGBA', (bigside, bigside), (255, 255, 255, 255))
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2), 0)))

        background.paste(image, offset)
        return background
    return image


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return resize(im.crop(bbox)).convert('L').resize((pic_sz, pic_sz))


class GwFontsDataset(Dataset):
    def __init__(
            self,
            train: bool,
            subset: list, #subset of the chars
            dataset_len: int,
            converted_dir
    ):
        self.directory = converted_dir + ("train" if train else "test")
        self.font_names = [dirname for dirname in os.listdir(self.directory) if dirname != '.DS_Store']
        self.font_len = len(self.font_names)
        self.train = train
        self.test = not self.train
        self.subset = [ord(i) for i in subset]

        self.alphabet_len = 2 * 26 + 10
        if self.test:
            self.train_directory = converted_dir + "train"
            self.train_names = [dirname for dirname in os.listdir(self.train_directory)][:self.nsamples]

    def __len__(self):
        return self.dataset_len


    def get_ordered_images(self, indx) -> list:
        if type(indx) == type("hello"):
            dirname = indx
            if self.train:            
                a = [Image.open(f"gwfonts/converted/train/{dirname}/{dirname}_{i}.png") for i in self.subset]
            else:
                a = [Image.open(f"gwfonts/converted/test/{dirname}/{dirname}_{i}.png") for i in self.subset]
            return a

        dirname = self.font_names[indx]
        if self.train:            
            a = [Image.open(f"gwfonts/converted/train/{dirname}/{dirname}_{i}.png") for i in self.subset]
        else:
            a = [Image.open(f"gwfonts/converted/test/{dirname}/{dirname}_{i}.png") for i in self.subset]
        return a
         

    @staticmethod
    def num_to_ord(idx):
        if idx < 10:
            return ord('0') + idx
        if idx < 36:
            return ord('A') + idx - 10
        return ord('a') + idx - 36

    
    def __getitem__(self, idx):
             
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((pic_sz, pic_sz)),
        ])
        pics = self.get_ordered_images(idx)
        return torch.stack(list(map(lambda x: transform(x), pics)))
            

def create_gw_data(subset:list):
    return GwFontsDataset(train=True, subset=subset, dataset_len=800, converted_dir="gwfonts/converted/")


def create_gw_loaders(nsamples=10, batch_size=4):
    data = create_gw_data(nsamples)

    return DataLoader(data[0], batch_size=batch_size, shuffle=True), \
        DataLoader(data[1], batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    train, test = create_gw_data()
    