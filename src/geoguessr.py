import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from transformers import BeitImageProcessor, BeitForImageClassification

import numpy as np
from tqdm import tqdm

from dataset import SingleImageDataset, SingleImageWithGridDataset
from Haversine import HaversineLoss

torch.manual_seed(42)

MEAN_LAT = 45.65404757
MEAN_LON = 7.95102084
STD_LAT = 4.644684882
STD_LON = 9.326391596

def standardizeLabels(labels):
    latitudes = labels[:, 0]
    longtitudes = labels[:, 1]

    latitudes = (latitudes - MEAN_LAT) / STD_LAT
    longtitudes = (longtitudes - MEAN_LON) / STD_LON

    labels[:, 0] = latitudes
    labels[:, 1] = longtitudes

    return labels


def unstandardizeLabels(labels):
    latitudes = labels[:, 0]
    longtitudes = labels[:, 1]

    latitudes = latitudes * STD_LAT + MEAN_LAT
    longtitudes = longtitudes * STD_LON + MEAN_LON

    labels[:, 0] = latitudes
    labels[:, 1] = longtitudes

    return labels

class Combined_Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid_classifier = nn.Linear(768,88)
        self.fc1 = nn.Linear(88 + 768 , 250)
        self.fc2 = nn.Linear(250, 150)
        self.fc3 = nn.Linear(150, 100)
        self.fc4 = nn.Linear(100, 80)
        self.fc5 = nn.Linear(80,50)
        self.output = nn.Linear(50,2)

    def forward(self, x):
        embeddings = x.clone()
        self.grid_output = self.grid_classifier(x)
        x = self.grid_output
        regressor_in = torch.cat([x,embeddings], dim =1)
        x = F.leaky_relu(self.fc1(regressor_in))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = self.output(x)
        return x

def main(annotatation_path, img_dir):
    if (annotatation_path is None or img_dir is None):
        dataset = SingleImageWithGridDataset()
    else:
        dataset = SingleImageWithGridDataset(annotatation_path, img_dir)

    train_set, val_set, _ = random_split(dataset, [0.6, 0.2, 0.2])

    train_loader = DataLoader(train_set, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)

    feature_extractor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-384')
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-384').to("cuda")

    model.classifier = Combined_Predictor().to("cuda")

    alpha = 0.5
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer_transformer = torch.optim.Adam(model.base_model.parameters(), lr=1e-5)
    optimizer_linear = torch.optim.Adam(model.classifier.parameters(), lr=2e-4)

    epochs = 100
    train_losses = []
    train_haversine = []
    val_losses = []
    val_haversine = []

    best_val_loss = float('inf')
    patience = 5
    counter = 0  

    haversine_metric = HaversineLoss()

    for epoch in tqdm(range(1, epochs+1)):
        running_train_loss = []
        running_val_loss = []
        running_train_haversine = []
        running_val_haversine = []

        for images, labels in train_loader:
            model.train()
            # labels = labels.cuda(non_blocking=True)
            labels_coord = standardizeLabels(labels[:,:2]).cuda(non_blocking=True)
            labels_grid_id = labels[:,2].long().cuda(non_blocking=True)
            
            optimizer_transformer.zero_grad()
            optimizer_linear.zero_grad()

            features = feature_extractor(images, return_tensors="pt")
            features = features['pixel_values'].cuda(non_blocking=True)

            y_pred = model(features)

            loss1 = criterion1(y_pred.logits, labels_coord)
            loss2 = criterion2(model.classifier.grid_output, labels_grid_id)
            loss = alpha * loss1 + (1-alpha)*loss2
            loss.backward()
            running_train_loss.append(loss.item())

            y_pred_rescaled = unstandardizeLabels(y_pred.logits.detach())
            running_train_haversine.append(haversine_metric(y_pred_rescaled, labels[:,:2]).item())

            optimizer_transformer.step()
            optimizer_linear.step()
        
        train_loss = np.mean(running_train_loss)
        train_losses.append(train_loss)
        train_haversine.append(np.mean(running_train_haversine))

        for images, labels in val_loader:
            with torch.no_grad():
                model.eval()
                labels = labels.cuda(non_blocking=True)
                labels = standardizeLabels(labels)

                features = feature_extractor(images, return_tensors="pt")
                features = features['pixel_values'].cuda(non_blocking=True)

                y_pred = model(features)

                loss1 = criterion1(y_pred.logits, labels_coord)
                loss2 = criterion2(model.classifier.grid_output, labels_grid_id)
                loss = alpha * loss1 + (1-alpha)*loss2
                running_val_loss.append(loss.item())

                running_val_haversine.append(haversine_metric(y_pred_rescaled, labels[:,:2]).item())
                y_pred_rescaled = unstandardizeLabels(y_pred.logits.detach())

        val_loss = np.mean(running_val_loss)
        val_losses.append(val_loss)
        val_haversine.append(np.mean(running_val_haversine))

        print(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_haversine': train_haversine,
            'val_haversine': val_haversine
        }, f'models/model_epoch_{epoch}.pt')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train your model.')

    # Add arguments
    parser.add_argument('--annotation_path', type=str, default=None, help='Path to the annotation file')
    parser.add_argument('--img_dir', type=str, default=None, help='Path to the directory containing images')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.annotation_path, args.img_dir)
