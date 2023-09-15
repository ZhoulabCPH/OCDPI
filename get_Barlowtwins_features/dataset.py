import os
import random

import openpyxl
import pandas as pd
import tables

from PIL import Image, ImageOps, ImageFilter
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class Transform_:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        return y1


class OV(Dataset):
    def __init__(self, store_dir, transform=None,
                 ):
        super(OV, self).__init__()
        self.store = tables.open_file(store_dir, mode='r')
        self.patches = self.store.root.patches_20
        self.images=[]
        self.transform = transform
        for i in range(self.patches.shape[0]):
            self.images.append(Image.fromarray(self.patches[i]))

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, item):
        image = self.images[item]
        label = ''
        image1, image2 = self.transform(image)
        return (image1, image2), label


class OV_(Dataset):
    def __init__(self, store_dir, label_dir,transform=None,
                 ):
        super(OV_, self).__init__()
        self.labels = pd.read_csv(label_dir)
        self.store = tables.open_file(store_dir, mode='r')
        self.patches = self.store.root.patches_80
        self.transform = transform

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, item):
        image = Image.fromarray(self.patches[item])
        label = self.labels.iloc[item, 1]
        image = self.transform(image)
        return image, label


def load_excel(excel_dir):
    wb = openpyxl.load_workbook(excel_dir)
    sheet = wb['Sheet1']
    rows = sheet.rows
    patches = []
    for i, row in enumerate(rows):
        patch = sheet['A' + str(i + 1)].value
        patches.append(patch)
    return patches


def load_image(image, image_path):
    image_dir = os.path.join(image_path, image)
    return Image.open(image_dir).convert('RGB')



