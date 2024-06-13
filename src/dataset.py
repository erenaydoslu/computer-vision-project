import os

import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image
import random
from typing import Tuple

torchvision.disable_beta_transforms_warning()

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
        self.annotations_file_path = annotations_file
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

def stratifiedSplit(dataset: FullLocationWithGridDataset, split_ratios: Tuple[float, float, float]):
    train, val, test = split_ratios
    sample_info = pd.read_csv(dataset.annotations_file_path)
    sample_info.reset_index(drop=True)
    tile_ids = sample_info["square_id"].unique()
    train_ids, val_ids, test_ids = [], [], []
    for id in tile_ids:
        samples_in_tile = sample_info[sample_info["square_id"] == id]
        indices_in_tile = samples_in_tile.index.to_list()
        random.shuffle(indices_in_tile)
        train_count = int(len(indices_in_tile) * train)
        val_count = int(len(indices_in_tile) * val)
        train_ids = train_ids + indices_in_tile[:train_count]
        val_ids = val_ids + indices_in_tile[train_count: train_count + val_count]
        test_ids = test_ids + indices_in_tile[train_count + val_count:]
    assert len(train_ids + val_ids + test_ids) == len(sample_info.index)
    for elem in train_ids:
        assert elem not in val_ids
        assert elem not in test_ids
    train_set = Subset(dataset, train_ids)
    val_set = Subset(dataset, val_ids)
    test_set = Subset(dataset, test_ids)
    return train_set, val_set, test_set
        
def calculateStats(dataloader):
    n = 0
    sum_lon = 0
    sum_lat = 0
    sum_sq_lon = 0
    sum_sq_lat = 0

    for _, labels in dataloader:
        latitudes = labels[:, 0].detach()
        longtitudes = labels[:, 1].detach()

        n += latitudes.nelement()
        sum_lon += longtitudes.sum().item()
        sum_lat += latitudes.sum().item()
        # sum_sq_lon += torch.pow(longtitudes, 2).sum().item()
        # sum_sq_lat += torch.pow(latitudes, 2).sum().item()
    mean_lon = sum_lon / n
    mean_lat = sum_lat / n
    for _, labels in dataloader:
        latitudes = labels[:, 0].detach()
        longtitudes = labels[:, 1].detach()

        sum_sq_lon += torch.pow(longtitudes - mean_lon, 2).sum().item()
        sum_sq_lat += torch.pow(latitudes - mean_lat, 2).sum().item()

    var_lon = sum_sq_lon / (n - 1)
    var_lat = sum_sq_lat / (n - 1)

    return mean_lon, mean_lat, var_lon**0.5, var_lat**0.5