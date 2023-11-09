import numpy as np
from torchvision.io import read_image
from torchvision.io import write_png
import os
import pathlib
import torch
from models.SeparatingStyleAndContent.model import SeparatingStyleAndContent as Model
from models.SeparatingStyleAndContent.ttf_to_png import TAGS
import numpy.random as npr


def style_input(folder, n_samples):
    folder: pathlib.Path = pathlib.Path(folder) if type(folder) == str else folder

    return torch.stack([read_image((folder / img).as_posix()) for img in os.listdir(folder)
                        if img[-4:] == ".png"][:n_samples]).to(dtype=float) / 255.


def generate_dicts(dataset_dir: pathlib.Path | str):
    if type(dataset_dir) == str:
        dataset_dir = pathlib.Path(dataset_dir)

    all_typefaces = [typeface for typeface in os.listdir(dataset_dir / "train")]

    tag_to_indices = {
        tag: [
            index for index, typeface in enumerate(all_typefaces) if tag in typeface.split('_')[-1]
        ] for tag in TAGS
    }

    return all_typefaces, tag_to_indices, dataset_dir / "train"


def generate_refs(tag: str, unicode: int, n_samples: int, all_typefaces, tag_to_indices, dataset_dir: pathlib.Path):
    refs_indices = npr.choice(tag_to_indices[tag], size=n_samples, replace=False)
    return torch.stack([read_image(
        (dataset_dir / all_typefaces[tp_i] / f"{all_typefaces[tp_i]}_{unicode}.png").as_posix()
    ) for tp_i in refs_indices]).to(dtype=float) / 255.


if __name__ == "__main__":
    N_SAMPLES = 5
    INPUT_FOLDER = "E:/INPUT_FOLDER"
    OUTPUT_FOLDER = "E:/OUTPUT_FOLDER"
    model: Model = torch.load("E:/model-cpu.pth")
    model.eval()
    names = []
    contents = []
    styles = []
    style = style_input(INPUT_FOLDER, n_samples=N_SAMPLES)
    dicts = generate_dicts("E:/google_fonts/converted")
    for tag in TAGS:
        for unicode in TAGS[tag]:
            names.append(f"{unicode}.png")
            contents.append(generate_refs(tag, unicode, N_SAMPLES, *dicts))
            styles.append(style.clone().detach())

    styles = torch.stack(styles)
    contents = torch.stack(contents)

    outputs = (model(contents, styles) * 255.).to(dtype=torch.uint8)

    for index, unicode in enumerate(names):
        write_png(outputs[index], OUTPUT_FOLDER + f"/{unicode}.png")








