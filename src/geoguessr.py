import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import BeitForImageClassification, BeitImageProcessor

from dataset import FullLocationWithGridDataset
from Haversine import HaversineLoss

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

torch.manual_seed(42)

MEAN_LAT = 45.6653417
MEAN_LON = 7.9323727
STD_LAT = 4.6619749
STD_LON = 9.3488671


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
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()
        self.grid_classifier = nn.Linear(768, 88)
        self.fc1 = nn.Linear(88 + 768, 250)
        # self.fc2 = nn.Linear(250, 150)
        # self.fc3 = nn.Linear(150, 100)
        # self.fc4 = nn.Linear(100, 80)
        self.fc5 = nn.Linear(250, 50)
        self.output = nn.Linear(50, 2)

    def forward(self, x):
        embeddings = x.clone()
        self.grid_output = self.grid_classifier(x)

        x = torch.nn.functional.softmax(self.grid_output, dim=1)
        regressor_in = torch.cat([x, embeddings], dim=1)

        x = self.relu(self.dropout((self.fc1(regressor_in))))
        # x = self.relu(self.dropout((self.fc2(x))))
        # x = self.relu(self.dropout((self.fc3(x))))
        # x = self.relu(self.dropout((self.fc4(x))))
        # x = self.relu((self.fc2(x)))
        # x = self.relu((self.fc3(x)))
        # x = self.relu((self.fc4(x)))
        x = self.relu(self.dropout((self.fc5(x))))
        x = self.output(x)
        return x

no_tqdm=True

def main(annotatation_path: str, img_dir: str):
    sample_type = annotatation_path[:-4].split("_")[-1]

    if annotatation_path is None or img_dir is None:
        dataset = FullLocationWithGridDataset()
    else:
        dataset = FullLocationWithGridDataset(annotatation_path, img_dir)

    train_set, val_set, _ = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(
        train_set, batch_size=32, num_workers=4, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=32, num_workers=4, shuffle=True, pin_memory=True
    )

    feature_extractor = BeitImageProcessor.from_pretrained(
        "microsoft/beit-base-patch16-384"
    )
    model = BeitForImageClassification.from_pretrained(
        "microsoft/beit-base-patch16-384"
    ).to("cuda")

    model.classifier = Combined_Predictor().to("cuda")

    alpha = 0.3
    criterion1 = nn.L1Loss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer_transformer = torch.optim.Adam(
        model.base_model.parameters(), lr=1e-5, weight_decay=0.01
    )
    optimizer_linear = torch.optim.Adam(
        model.classifier.parameters(), lr=2e-4, weight_decay=0.01
    )
    scheduler_transformer = ReduceLROnPlateau(
        optimizer_transformer, factor=0.5, patience=2
    )
    scheduler_linear = ReduceLROnPlateau(optimizer_linear, factor=0.5, patience=2)

    epochs = 100
    train_losses = []
    train_class_losses = []
    train_coord_losses = []
    train_haversine = []
    val_losses = []
    val_class_losses = []
    val_coord_losses = []
    val_haversine = []

    best_val_loss = float("inf")
    patience = 5
    counter = 0

    haversine_metric = HaversineLoss()

    for epoch in tqdm(range(1, epochs + 1), disable=no_tqdm):
        running_train_loss = []
        class_train_loss = []
        coord_train_loss = []
        running_val_loss = []
        class_val_loss = []
        coord_val_loss = []
        running_train_haversine = []
        running_val_haversine = []

        for images, labels in tqdm(
            train_loader, desc="TRAIN", total=len(train_set) / train_loader.batch_size, disable=no_tqdm
        ):
            model.train()
            labels = labels.cuda(non_blocking=True)
            labels_coord = standardizeLabels(labels[:, :2]).cuda(non_blocking=True)
            labels_grid_id = labels[:, 2].long().cuda(non_blocking=True)

            optimizer_transformer.zero_grad()
            optimizer_linear.zero_grad()

            features = feature_extractor(images, return_tensors="pt")
            features = features["pixel_values"].cuda(non_blocking=True)

            y_pred = model(features)

            loss1 = criterion1(y_pred.logits, labels_coord)
            loss2 = criterion2(model.classifier.grid_output, labels_grid_id)
            loss = alpha * loss1 + (1 - alpha) * loss2
            loss.backward()
            running_train_loss.append(loss.item())
            coord_train_loss.append(loss1.item())
            class_train_loss.append(loss2.item())

            y_pred_rescaled = unstandardizeLabels(y_pred.logits.detach())
            running_train_haversine.append(
                haversine_metric(y_pred_rescaled, labels[:, :2]).item()
            )

            optimizer_transformer.step()
            optimizer_linear.step()

        train_loss = np.mean(running_train_loss)
        train_coord_loss = np.mean(coord_train_loss)
        train_class_loss = np.mean(class_train_loss)
        train_losses.append(train_loss)
        train_coord_losses.append(train_coord_loss)
        train_class_losses.append(train_class_loss)
        train_haversine.append(np.mean(running_train_haversine))

        for images, labels in tqdm(
            val_loader, desc="VAL", total=len(val_set) / val_loader.batch_size, disable=no_tqdm
        ):
            with torch.no_grad():
                model.eval()
                labels = labels.cuda(non_blocking=True)
                labels_coord = standardizeLabels(labels[:, :2]).cuda(non_blocking=True)
                labels_grid_id = labels[:, 2].long().cuda(non_blocking=True)

                features = feature_extractor(images, return_tensors="pt")
                features = features["pixel_values"].cuda(non_blocking=True)

                y_pred = model(features)

                loss1 = criterion1(y_pred.logits, labels_coord)
                loss2 = criterion2(model.classifier.grid_output, labels_grid_id)
                loss = alpha * loss1 + (1 - alpha) * loss2
                running_val_loss.append(loss.item())
                coord_val_loss.append(loss1.item())
                class_val_loss.append(loss2.item())

                y_pred_rescaled = unstandardizeLabels(y_pred.logits.detach())
                running_val_haversine.append(
                    haversine_metric(y_pred_rescaled, labels[:, :2]).item()
                )

        val_loss = np.mean(running_val_loss)
        val_coord_loss = np.mean(coord_val_loss)
        val_class_loss = np.mean(class_val_loss)
        val_losses.append(val_loss)
        val_coord_losses.append(val_coord_loss)
        val_class_losses.append(val_class_loss)
        val_haversine.append(np.mean(running_val_haversine))

        print(
            f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}"
        )
        print(
            f"Grid Class:    Train Loss: {train_class_loss:.2f}, Val Loss: {val_class_loss:.2f}"
        )
        print(
            f"Coordinates:   Train Loss: {train_coord_loss:.2f}, Val Loss: {val_coord_loss:.2f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "train_class_losses": train_class_losses,
                "train_coord_losses": train_coord_losses,
                "val_losses": val_losses,
                "val_class_losses": val_class_losses,
                "val_coord_losses": val_coord_losses,
                "train_haversine": train_haversine,
                "val_haversine": val_haversine,
            },
            f"models/model_{sample_type}_epoch_{epoch}.pt",
        )

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
        scheduler_linear.step(val_loss)
        scheduler_transformer.step(val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your model.")
    torch.cuda.empty_cache()

    # Add arguments
    parser.add_argument(
        "--annotation_path", type=str, default=None, help="Path to the annotation file"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="Path to the directory containing images",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.annotation_path, args.img_dir)
