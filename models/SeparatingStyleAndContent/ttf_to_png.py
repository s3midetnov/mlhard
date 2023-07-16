import os
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import numpy.random as npr
import shutil
from fontTools.ttLib import TTFont

import typing as tp


def chr_range(c1, c2) -> list[int]:
    return list(range(ord(c1), ord(c2) + 1))


LATIN_LOWERCASE = chr_range('a', 'z')
LATIN_UPPERCASE = chr_range('A', 'Z')
DIGITS = chr_range('0', '9')
CYRILLIC_UPPERCASE = list(range(1040, 1072))
CYRILLIC_LOWERCASE = list(range(1072, 1104))
GREEK_UPPERCASE = list(filter(lambda x: x != 930, range(913, 938)))
GREEK_LOWERCASE = list(range(945, 970))

TAGS = {
    'l': LATIN_LOWERCASE,
    'L': LATIN_UPPERCASE,
    'd': DIGITS,
    'C': CYRILLIC_UPPERCASE,
    'c': CYRILLIC_LOWERCASE,
    'G': GREEK_UPPERCASE,
    'g': GREEK_LOWERCASE
}


def check_font_contains(tt: TTFont, unicode_codes: tp.Iterable[int]) -> bool:
    """
    :param tt: any true type font
    :param unicode_codes: iterable of characters represented by their unicodes
    :return: checks if given font contains given set of characters
    """
    for char in unicode_codes:
        found = False
        for cmap in tt['cmap'].tables:
            if cmap.isUnicode():
                if char in cmap.cmap:
                    found = True
        if not found:
            return False
    return True


def folder_to_save(filename: str | Path, suffix: str = '', train=True, converted: str | Path | None = None):
    """
    :param filename: ttf file
    :param suffix: suffix which must be added to a folder containing pngs
    :param train: train or test folder to choose
    :param converted: path to save the folder
    """
    path = filename if type(filename) == Path else Path(filename)
    assert path.suffix == '.ttf'
    parent = path.parent

    if converted is None:
        converted_folder = parent / "converted"
    else:
        converted_folder = Path(converted) if type(converted) == str else converted

    if not converted_folder.is_dir():
        Path.mkdir(converted_folder)

    train_test_folder = converted_folder / ("train" if train else "test")
    if not train_test_folder.is_dir():
        Path.mkdir(train_test_folder)

    save_folder = train_test_folder / f"{path.name[:-4]}_{suffix}"
    if not save_folder.is_dir():
        Path.mkdir(save_folder)

    return save_folder


def ttf_to_pngs(filename: str | Path, train=True):
    """
    :param filename: source .ttf file
    :param train: choose train or test folder to put images to
    :return: makes a bunch of .png files from a single .ttf (0-9, a-z, A-Z)
    """
    path = filename if type(filename) == Path else Path(filename)
    assert path.suffix == ".ttf"
    save_folder = folder_to_save(path, train=train)

    adequate_chars = [chr(x) for x in chr_range('0', '9') + chr_range('a', 'z') + chr_range('A', 'Z')]

    for character in adequate_chars:
        image = Image.new(mode='L', size=(80, 80), color=256)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(path.as_posix(), size=70)
        _, _, x, y = draw.textbbox((0, 0), text=character, font=font)
        draw.text(xy=((80 - x) / 2, (80 - y) / 2), text=character, font=font)

        image.save(save_folder / f"{path.name[:-4]}_{ord(character)}.png")


def ttf_to_pngs_extended(filename: str | Path, train=True):
    res: str = ''

    path = filename if type(filename) == Path else Path(filename)
    assert path.suffix == '.ttf'

    tt = TTFont(str(filename))

    for tag, alphabet in TAGS.items():
        if check_font_contains(tt, alphabet):
            res += tag

    save_folder = folder_to_save(path, suffix=res, train=train)

    for tag in res:
        alphabet = TAGS[tag]
        for character in alphabet:
            image = Image.new(mode='L', size=(80, 80), color=256)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(path.as_posix(), size=60, encoding='unic')
            _, _, right, bottom = font.getbbox(chr(character), stroke_width=0)
            draw.text(xy=((80 - right) / 2, (80 - bottom) / 2), text=chr(character), font=font)
            image.save(save_folder / f"{path.name[:-4]}_{res}_{character}.png")

    return res


def make_dataset_from_ttf_list(folder: str | Path, ptest=0.2, extended = False):
    """
    :param extended: use extended version of dataset or not
    :param folder: source folder
    :param ptest: probability of a single typeface to end up in test part of the dataset
    :return: Makes a dataset divided in train and test parts, each consisting from
    several typefaces. Each typeface is represented by 0-9, a-z, A-Z.
    """
    path = folder if type(folder) == Path else Path(folder)

    if not path.is_dir():
        print(f"'{path}' is not a directory. Terminated...")
        return

    train = 0
    test = 0

    res_dict = {tag: 0 for tag in TAGS}
    for file in os.listdir(path):
        filepath = path / file
        if filepath.suffix == ".ttf":
            train_mark = npr.random() > ptest
            try:
                if not extended:
                    ttf_to_pngs(filepath, train_mark)
                else:
                    res = ttf_to_pngs_extended(filepath, train_mark)
                    for tag in res:
                        res_dict[tag] += 1
                if train_mark:
                    train += 1
                else:
                    test += 1
            except OSError:
                print(f"Failed {filepath}")

    print(f"Train {train}, Test {test}")
    print(res_dict)


def extract_all_ttf_from_folder(folder: str | Path, target_folder: str):
    """
    :param folder: source folder
    :param target_folder: destination folder
    :return: copies all .ttf files from source folder to destination folder recursively
    """
    if type(folder) == str:
        path = Path(folder)
    else:
        path = folder
    if not path.is_dir():
        print(f"'{path}' is not a directory. Terminated...")
        return

    success = 0
    failure = 0

    for file in os.listdir(path):
        filepath = path / file
        if filepath.is_dir():
            extract_all_ttf_from_folder(filepath, target_folder)
        elif filepath.suffix == '.ttf':
            try:
                shutil.copy(filepath, target_folder)
                success += 1
            except (PermissionError, OSError):
                failure += 1


def check_completeness(folder: str, extended: bool = False):
    """
    :param extended: use not only latin letters but also greek and cyrillic
    :param folder: folder to check
    :return: checks the folder for completeness (each typeface must contain exactly 62 characters)
    """
    path = Path(folder)
    train = path / "train"
    test = path / "test"

    def check(p: Path):
        for typeface in os.listdir(p):
            cnt = 0
            for _ in os.listdir(p / typeface):
                cnt += 1

            total = 62 if not extended else sum(len(TAGS[tag]) for tag in  typeface.split('_')[-1])
            if cnt != total:
                print(typeface)

    check(train)
    check(test)


if __name__ == "__main__":
    # make_dataset_from_ttf_list('E:/google_fonts', extended=True)
    check_completeness('E:/google_fonts/converted', extended=True)
    check_completeness('../../datasets/converted_gwfonts')
