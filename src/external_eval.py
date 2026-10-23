
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, classification_report

from data_utils import (
    PROJECT_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    IMAGE_SIZE,
    load_classes
)


EXTERNAL_DIR = PROJECT_DIR / "data" / "external_images"
LABELS_FILE = PROJECT_DIR / "data" / "external_labels.csv"


class ExternalPetDataset(Dataset):

    def __init__(self, dataframe, class_to_idx):
        self.data = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx

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

        image_path = EXTERNAL_DIR / row["filename"]

        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))

        label = self.class_to_idx[row["label"]]

        return image, label


def main():
    model_path = MODEL_DIR / "pet_breed_resnet18.pth"

    if not model_path.is_file():
        print("Train the model first using: python src/train.py")
        return

    if not LABELS_FILE.is_file():
        print("Missing file:", LABELS_FILE)
        return

    df = pd.read_csv(LABELS_FILE)

    required_columns = {"filename", "label", "source_url"}

    if not required_columns.issubset(df.columns):
        print("CSV columns must be: filename,label,source_url")
        return

    if len(df) < 30:
        print("Please add at least 30 external images.")
        print("Current image count:", len(df))
        return

    if df["filename"].duplicated().any():
        print("Please remove duplicate filenames from the CSV.")
        return

    missing = [
        name for name in df["filename"]
        if not (EXTERNAL_DIR / name).is_file()
    ]

    if missing:
        print("Some external image files are missing.")
        print("First missing file:", missing[0])
        return

    if df["source_url"].isna().any() or (
        df["source_url"].astype(str).str.strip() == ""
    ).any():
        print("Please provide a source URL for every image.")
        return

    classes, class_to_idx = load_classes()

    unknown = sorted(set(df["label"]) - set(classes))

    if unknown:
        print("Unknown breed labels:", unknown)
        print("Use exact labels from results/classes.json.")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    dataset = ExternalPetDataset(df, class_to_idx)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    true_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))

            predictions.extend(
                outputs.argmax(1).cpu().tolist()
            )

            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)

    print("\nEXTERNAL EVALUATION")
    print("-------------------")
    print("External images:", len(df))
    print(f"External accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(
        true_labels,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    ))

    result = df.copy()

    result["true_label"] = [
        classes[index] for index in true_labels
    ]

    result["predicted_label"] = [
        classes[index] for index in predictions
    ]

    result["correct"] = (
        result["true_label"] == result["predicted_label"]
    )

    output_path = RESULTS_DIR / "external_predictions.csv"
    result.to_csv(output_path, index=False)

    print("External predictions saved to:", output_path)


if __name__ == "__main__":
    main()
