import os
from glob import glob

import PIL
from PIL import Image
from PIL import ImageFile
import warnings

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


def picture_size():
    img_path = "../data/train/"
    img_list1 = glob(os.path.join(img_path, "*", "*.jpg"))
    img_list1 += glob(os.path.join(img_path, "*", "*.png"))
    width1 = 0
    height1 = 0
    for idx in range(len(img_list1)):
        img = PIL.Image.open(img_list1[idx])
        width1 += img.size[0]
        height1 += img.size[1]

    img_path = "../data/test/"
    img_list2 = glob(os.path.join(img_path, "*.jpg"))
    img_list2 += glob(os.path.join(img_path, "*.png"))
    width2 = 0
    height2 = 0
    for idx in range(len(img_list2)):
        img = PIL.Image.open(img_list2[idx])
        width2 += img.size[0]
        height2 += img.size[1]

    print("train: (%d, %d), test: (%d, %d), all: (%d, %d)" % (
        width1 / len(img_list1), height1 / len(img_list1),
        width2 / len(img_list2), height2 / len(img_list2),
        (width1 + width2) / (len(img_list1) + len(img_list2)), (height1 + height2) / (len(img_list1) + len(img_list2)),
    ))


if __name__ == '__main__':
    picture_size()
