from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from models.SeparatingStyleAndContent.ttf_to_png import TAGS
import os
import numpy as np
import numpy.random as npr

import typing as tp

from torchvision.io import read_image
import torch


def compare_tags(c: str, d: str):
    return TAGS[c][0] < TAGS[d][0]


class GoogleDataset(Dataset):
    def __init__(self, nsamples, dataset_len, directory: str | Path, train: bool):
        self.nsamples = nsamples
        self.dataset_len = dataset_len
        self.train = train
        self.directory = (Path(directory) if type(directory) == str else directory) / ("train" if train else "test")
        self.all_typefaces = [typeface for typeface in os.listdir(self.directory)]
        typefaces_len: int = len(self.all_typefaces)

        tag_to_indices = {
            tag: [
                index for index, typeface in enumerate(self.all_typefaces) if tag in typeface.split('_')[-1]
            ] for tag in TAGS
        }
        self.coord: list[tuple[int, int, int, str]] = []
        typefaces_size = [0 for _ in range(typefaces_len)]
        for ti, typeface in enumerate(self.all_typefaces):
            for gi, glyph in enumerate(os.listdir(self.directory / typeface)):
                unicode = int(glyph[:-4].split('_')[-1])
                self.coord.append((ti, gi, unicode, self.return_tag(unicode)))
                typefaces_size[ti] += 1
        self.glyphs_len: int = len(self.coord)

        self.targets = npr.choice(a=self.glyphs_len, size=self.dataset_len, replace=self.glyphs_len >= self.dataset_len)
        self.style_refs: tp.List[tp.Optional[np.ndarray]] = [None for _ in range(dataset_len)]
        self.content_refs: tp.List[tp.Optional[np.ndarray]] = [None for _ in range(dataset_len)]
        for i in range(self.dataset_len):
            ti, gi, unicode, tag = self.coord[self.targets[i]]

            _content_refs = npr.choice(a=len(tag_to_indices[tag]) - 1, size=self.nsamples, replace=False)

            coincident_ti = None
            for j in range(self.nsamples):
                if tag_to_indices[tag][_content_refs[j]] == ti:
                    coincident_ti = j

            if coincident_ti is not None:
                _content_refs += _content_refs >= _content_refs[coincident_ti]

            for j in range(self.nsamples):
                _content_refs[j] = tag_to_indices[tag][_content_refs[j]]

            self.content_refs[i] = _content_refs

            _style_refs = npr.choice(a=typefaces_size[ti] - 1, size=self.nsamples, replace=False)
            _style_refs = _style_refs + (_style_refs >= gi)

            whole_tag = self.all_typefaces[ti].split('_')[-1]
            for j in range(self.nsamples):
                _style_refs[j] = self.return_unicode(whole_tag, _style_refs[j])
            self.style_refs[i] = _style_refs

    def __getitem__(self, idx):
        ti, gi, unicode, tags = self.coord[self.targets[idx]]
        tp_name = self.all_typefaces[ti]
        style_coords = self.style_refs[idx]  # unicodes
        tp_names = [self.all_typefaces[self.content_refs[idx][i]] for i in range(self.nsamples)]

        def construct_path(typeface_name, unicode_int):
            return (self.directory / typeface_name / f"{typeface_name}_{unicode_int}.png").as_posix()

        return (torch.stack([
            read_image(construct_path(tp_names[r], unicode)) for r in range(self.nsamples)
        ]).to(dtype=float) / 255., torch.stack([
            read_image(construct_path(tp_name, style_coords[r])) for r in range(self.nsamples)
        ]).to(dtype=float) / 255.),\
            read_image(construct_path(tp_name, unicode)).to(dtype=float) / 255.

    @staticmethod
    def return_unicode(tags: str, order: int):
        sorted_tags = list(sorted(tags, key=lambda t: TAGS[t][0]))
        copy_order = order
        for tag in sorted_tags:
            if len(TAGS[tag]) <= copy_order:
                copy_order -= len(TAGS[tag])
            else:
                return TAGS[tag][0] + copy_order + (1 if copy_order > 16 and tag == 'G' else 0)
        print(tags, order)
        print(sorted_tags)

    @staticmethod
    def return_order(tags: str, unicode: int):
        unicode_tag = GoogleDataset.return_tag(unicode)
        order = 0
        for tag in tags:
            if compare_tags(unicode_tag, tag):
                order += len(TAGS[tag])
        order += unicode - TAGS[unicode_tag][0]
        return order - (TAGS['G'][0] <= unicode <= TAGS['G'][-1])

    @staticmethod
    def return_tag(x: int):
        if x < 58:
            return 'd'
        if x < 91:
            return 'L'
        if x < 123:
            return 'l'
        if x < 938:
            return 'G'
        if x < 970:
            return 'g'
        if x < 1072:
            return 'C'
        if x < 1104:
            return 'c'

    def __len__(self):
        return self.dataset_len


def create_google_datasets(directory: str, nsamples=10, dataset_lens=(10000, 2000)):
    return GoogleDataset(nsamples, dataset_lens[0], directory, train=True), \
            GoogleDataset(nsamples, dataset_lens[1], directory, train=False)


def create_google_dataloaders(directory: str, nsamples=10, batch_size=4, dataset_lens=(10000, 2000)):
    data = create_google_datasets(directory, nsamples, dataset_lens)
    return DataLoader(data[0], batch_size=batch_size, shuffle=True), \
        DataLoader(data[1], batch_size=batch_size, shuffle=False)
