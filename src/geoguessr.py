import argparse

import torch
from torch.utils.data import random_split, DataLoader
from transformers import BeitFeatureExtractor, BeitForImageClassification

import numpy as np
from tqdm import tqdm

from dataset import SingleImageDataset
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
    longtitudes[:, 1] = longtitudes

    return labels


def main(annotatation_path, img_dir):
    if (annotatation_path is None or img_dir is None):
        dataset = SingleImageDataset()
    else:
        dataset = SingleImageDataset(annotatation_path, img_dir)

    train_set, val_set, _ = random_split(dataset, [0.6, 0.2, 0.2])

    train_loader = DataLoader(train_set, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)

    feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-384')
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-384').to("cuda")

    model.classifier = torch.nn.Linear(768, 2).to("cuda")

    criterion = torch.nn.MSELoss()
    optimizer_transformer = torch.optim.Adam(model.base_model.parameters(), lr=5e-6)
    optimizer_linear = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

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
            labels = labels.cuda(non_blocking=True)
            labels = standardizeLabels(labels)
            
            optimizer_transformer.zero_grad()
            optimizer_linear.zero_grad()

            features = feature_extractor(images, return_tensors="pt")
            features = features['pixel_values'].cuda(non_blocking=True)

            y_pred = model(features)

            loss = criterion(y_pred.logits, labels)
            loss.backward()
            running_train_loss.append(loss.item())

            y_pred, labels = unstandardizeLabels(y_pred.logits.detach()), unstandardizeLabels(labels)
            running_train_haversine.append(haversine_metric(y_pred, labels).item())

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

                loss = criterion(y_pred.logits, labels) 
                running_val_loss.append(loss.item())

                y_pred, labels = unstandardizeLabels(y_pred.logits.detach()), unstandardizeLabels(labels)
                running_val_haversine.append(haversine_metric(y_pred, labels).item())

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
