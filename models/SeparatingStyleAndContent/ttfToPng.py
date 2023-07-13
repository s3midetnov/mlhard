import os

from PIL import Image, ImageFont, ImageDraw
from pathlib import Path

import numpy.random as npr


def ttf_to_pngs(filename: str | Path, train=True):
    if type(filename) == str:
        path = Path(filename)
    else:
        path = filename

    assert path.suffix == ".ttf"

    parent = path.parent

    converted_folder = parent / "converted"
    if not converted_folder.is_dir():
        Path.mkdir(converted_folder)

    train_test_folder = converted_folder / ("train" if train else "test")
    if not train_test_folder.is_dir():
        Path.mkdir(train_test_folder)

    save_folder = train_test_folder / f"{path.name[:-4]}"
    if not save_folder.is_dir():
        Path.mkdir(save_folder)

    def chr_range(c1, c2):
        return list(range(ord(c1), ord(c2) + 1))

    adequate_chars = [chr(x) for x in chr_range('0', '9') + chr_range('a', 'z') + chr_range('A', 'Z')]

    for character in adequate_chars:
        image = Image.new(mode='L', size=(80, 80), color=256)
        draw = ImageDraw.Draw(image)

        # print(path.as_posix())
        font = ImageFont.truetype(path.as_posix(), size=70)

        _x, _y, x, y = draw.textbbox((0, 0), text=character, font=font)
        # print(_x, _y, x, y)
        draw.text(xy=((80 - x) / 2, (80 - y) / 2), text=character, font=font)

        image.save(save_folder / f"{path.name[:-4]}_{ord(character)}.png")


def make_gwpngs(ptest=0.2):
    path = Path("../../datasets/gwfonts")
    # print(path.as_posix())
    train = 0
    test = 0
    failure = 0
    for file in os.listdir(path):
        filepath = path / file
        if filepath.suffix == ".ttf":
            train_mark = npr.random() > ptest
            try:
                ttf_to_pngs(filepath, train_mark)
                if train_mark:
                    train += 1
                else:
                    test += 1
            except OSError:
                failure += 1
    print(f"Train {train}, Test {test}, Failure {failure}")


if __name__ == "__main__":
    make_gwpngs()
