from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from ttf_to_png import TAGS
import os
import numpy as np
import numpy.random as npr

import typing as tp


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
            ti, gi, unicode, tag = self.coord[i]

            _content_refs = npr.choice(a=tag_to_indices[tag], size=self.nsamples)
            self.content_refs[i] = _content_refs + _content_refs >= ti

            _style_refs = npr.choice(a=typefaces_size[i], size=self.nsamples)
            whole_tag = self.all_typefaces[ti].split('_')[-1]
            for j in range(self.nsamples):
                _style_refs[j] = self.return_unicode(whole_tag, _style_refs[j])
            self.style_refs[i] = _style_refs[i] + _style_refs[i]

    @staticmethod
    def return_unicode(tags: str, order: int):
        tcode = order
        if 'd' in tags:
            if tcode < 10: return 48 + tcode
            else: tcode -= 10
        if 'L' in tags:
            if tcode < 26: return 65 + tcode
            else: tcode -= 26
        if 'l' in tags:
            if tcode < 26: return 97 + tcode
            else: tcode -= 26
        if 'G' in tags:
            if tcode < 17: return 913 + tcode
            elif tcode < 24: return 913 + tcode + 1
            else: tcode -= 24
        if 'g' in tags:
            if tcode < 24: return 945 + tcode
            else: tcode -= 24
        if 'C' in tags:
            if tcode < 32: return 1040 + tcode
            else: tcode -= 32
        return 1072 + tcode


    @staticmethod
    def return_order(tags: str, unicode: int):
        tcode = 0
        if 'd' in tags:
            if unicode < 58: return unicode - 48 + tcode
            else: tcode += 10
        if 'L' in tags:
            if unicode < 91: return unicode - 65 + tcode
            else: tcode += 26
        if 'l' in tags:
            if unicode < 123: return unicode - 97 + tcode
            else: tcode += 26
        if 'G' in tags:
            if unicode < 930: return unicode - 913 + tcode
            elif unicode < 938: return unicode - 914 + tcode
            else: tcode += 24
        if 'g' in tags:
            if unicode < 970: return unicode - 945 + tcode
            else: tcode += 24
        if 'C' in tags:
            if unicode < 1072: return unicode - 1040 + tcode
            else: tcode += 32
        return unicode - 1072 + tcode

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

    def __getitem__(self, index):
        pass
