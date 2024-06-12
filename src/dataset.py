import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

# #Single sample is a single image from a random location
# class SingleImageDataset(Dataset):
#     def __init__(self, annotations_file="../data/annotations_unglued.csv", img_dir="../data/images", transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = torch.Tensor([self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 3]])
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

# #Single sample is a single image from a random location
# class SingleImageWithGridDataset(Dataset):
#     def __init__(self, annotations_file="../data/annotations_unglued.csv", img_dir="../data/images", transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = torch.Tensor([self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 3], self.img_labels.iloc[idx, 1]])
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label


# Single sample is all images from a given (random) location
class FullLocationWithGridDataset(Dataset):
    def __init__(
        self,
        annotations_file="../data/annotations_glued.csv",
        img_dir="../data/images",
        transform=None,
        target_transform=None,
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
        # return 1000

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        images = read_image(img_path)

        label = torch.Tensor(
            [
                self.img_labels.iloc[idx, 2],
                self.img_labels.iloc[idx, 3],
                self.img_labels.iloc[idx, 1],
            ]
        )
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            label = self.target_transform(label)
        return images, label
