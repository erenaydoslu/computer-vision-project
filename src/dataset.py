import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset


#Single sample is a single image from a random location
class SingleImageDataset(Dataset):
    def __init__(self, annotations_file="../data/annotations_unglued.csv", img_dir="../data/images", transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.Tensor([self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 3]])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#Single sample is a single image from a random location
class SingleImageWithGridDataset(Dataset):
    def __init__(self, annotations_file="../data/annotations_unglued.csv", img_dir="../data/images", transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.Tensor([self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 3], self.img_labels.iloc[idx, 1]])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

    #Single sample is all images from a given (random) location
class FullLocationDataset(Dataset):
    def __init__(self, annotations_file="../data/annotations_unglued.csv", img_dir="../data/images", transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return int(len(self.img_labels) // 3)

    def __getitem__(self, idx):
        loc_idx = (idx // 3) * 3
        
        img_paths = [os.path.join(self.img_dir, self.img_labels.iloc[loc_idx + i, 0]) for i in range(3)]
        images = torch.stack([read_image(img_path) for img_path in img_paths], dim=0)
        labels = torch.Tensor([self.img_labels.iloc[loc_idx, 2], self.img_labels.iloc[loc_idx, 3]])
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            labels = self.target_transform(labels)
        return images, labels