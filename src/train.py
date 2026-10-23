
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from data_utils import (
    RESULTS_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    PetDataset,
    load_classes
)


SEED = 42
EPOCHS = 8
LEARNING_RATE = 0.0001
NUM_WORKERS = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    if not (RESULTS_DIR / "train.csv").is_file():
        print("Please run: python src/prepare_data.py")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    classes, class_to_idx = load_classes()

    print("Device:", device)
    print("Number of breeds:", len(classes))

    train_set = PetDataset(
        RESULTS_DIR / "train.csv",
        class_to_idx,
        training=True
    )

    val_set = PetDataset(
        RESULTS_DIR / "val.csv",
        class_to_idx,
        training=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Load a model pretrained on ImageNet
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception as error:
        print("Could not download pretrained model weights.")
        print("Check your internet connection and try again.")
        print("Error:", error)
        return

    # Replace the final layer for our 37 breeds
    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):

        # Training
        model.train()

        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

        # Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_function(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (
                    outputs.argmax(1) == labels
                ).sum().item()

        train_loss /= len(train_set)
        val_loss /= len(val_set)

        train_accuracy = train_correct / len(train_set)
        val_accuracy = val_correct / len(val_set)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
            f"Validation accuracy: {val_accuracy:.4f}"
        )

        # Save weights from the best validation epoch
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    save_path = MODEL_DIR / "pet_breed_resnet18.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "image_size": 224
    }, save_path)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
