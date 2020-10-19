import os
import random
import shutil
import warnings
from glob import glob

import PIL
import cv2
import torch
import torchvision
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from utils.transformer import *


warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BinaryClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, input_size, mode=None, transforms=None, argument_path=False):
        self.mode = mode
        if self.mode == 'test':
            self.img_list = glob(os.path.join(img_path, "*.jpg"))
            self.img_list += glob(os.path.join(img_path, "*.png"))
        else:
            self.img_list = glob(os.path.join(img_path, "*", "*.jpg"))
            self.img_list += glob(os.path.join(img_path, "*", "*.png"))
            if argument_path:
                self.img_list += glob(os.path.join("./data/old_train", "*", "*.jpg"))
        if transforms is None:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(size=input_size),  # 缩放
                Resize(size=input_size),  # 等比填充缩放
                # torchvision.transforms.RandomCrop(size=input_size),
                # torchvision.transforms.RandomResizedCrop(size=input_size, scale=(0.6, 1.0)),
                torchvision.transforms.RandomHorizontalFlip(),
                # RandomGaussianBlur(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            img = PIL.Image.open(self.img_list[idx])
            if img.format == 'PNG' or img.format == 'GIF':
                img = img.convert("RGB")
            elif img.layers == 1:
                img = cv2.imread(self.img_list[idx], 0)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = self.transforms(img)
        except OSError:
            print("OSError at combined_mask path ", self.img_list[idx])
            return None

        if self.mode == 'train' or self.mode == 'eval':
            if self.img_list[idx].split('/')[-2] == "phone":
                label = 2
            elif self.img_list[idx].split('/')[-2] == "smoke":
                label = 1
            elif self.img_list[idx].split('/')[-2] == "normal":
                label = 0
            elif self.img_list[idx].split('/')[-2] == "calling_images":
                label = 2
            elif self.img_list[idx].split('/')[-2] == "smoking_images":
                label = 1
            elif self.img_list[idx].split('/')[-2] == "normal_images":
                label = 0
            # elif self.img_list[idx].split('/')[-2] == "5k_cigarettes":
            #     label = 1
            else:
                print(self.img_list[idx].split('/')[-2])
                raise ValueError
            sample = {"image": img, "label": label, "name": self.img_list[idx].split('/')[-1]}
        else:
            sample = {"image": img, "name": self.img_list[idx].split('/')[-1]}

        return sample


def build_dataset_train(data_path, input_size):
    return BinaryClassifierDataset(os.path.join(data_path, "train"), input_size, mode='train')


def build_dataset_eval(data_path, input_size):
    return BinaryClassifierDataset(os.path.join(data_path, "eval"), input_size, mode='eval')


def build_dataset_test(data_path, input_size):
    return BinaryClassifierDataset(os.path.join(data_path, "test"), input_size, mode='test')


def split_old_dataset():
    train_path = "../data/train/"
    eval_path = "../data/eval/"
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    if not os.path.exists(eval_path + "calling_images/"):
        os.mkdir(eval_path + "calling_images/")
    if not os.path.exists(eval_path + "normal_images/"):
        os.mkdir(eval_path + "normal_images/")
    if not os.path.exists(eval_path + "smoking_images/"):
        os.mkdir(eval_path + "smoking_images/")

    for p in ["calling_images/", "normal_images/", "smoking_images/"]:
        img_list = glob(os.path.join(train_path + p, "*.jpg"))
        random.shuffle(img_list)
        img_list = img_list[:int(0.2 * len(img_list))]
        for img in img_list:
            new_img = img.replace("train", "eval")
            shutil.move(img, new_img)


def split_dataset():
    train_path = "../data/train/"
    eval_path = "../data/eval/"
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    if not os.path.exists(eval_path + "phone/"):
        os.mkdir(eval_path + "phone/")
    if not os.path.exists(eval_path + "normal/"):
        os.mkdir(eval_path + "normal/")
    if not os.path.exists(eval_path + "smoke/"):
        os.mkdir(eval_path + "smoke/")

    for p in ["phone/", "normal/", "smoke/"]:
        img_list = glob(os.path.join(train_path + p, "*.jpg"))
        img_list += glob(os.path.join(train_path + p, "*.png"))
        random.shuffle(img_list)
        img_list = img_list[:int(0.2 * len(img_list))]
        for img in img_list:
            new_img = img.replace("train", "eval")
            shutil.move(img, new_img)


if __name__ == '__main__':
    split_dataset()
