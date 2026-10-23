
from pathlib import Path
import json

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "oxford-iiit-pet"
IMAGE_DIR = DATA_DIR / "images"
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_DIR = PROJECT_DIR / "models"

IMAGE_SIZE = 224
BATCH_SIZE = 32


class PetDataset(Dataset):

    def __init__(self, csv_path, class_to_idx, training=False):
        self.data = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx

        if training:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = IMAGE_DIR / row["filename"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = self.class_to_idx[row["label"]]

        return image, label


def load_classes():
    classes_path = RESULTS_DIR / "classes.json"

    with open(classes_path, "r", encoding="utf-8") as file:
        classes = json.load(file)

    class_to_idx = {
        name: index for index, name in enumerate(classes)
    }

    return classes, class_to_idx
