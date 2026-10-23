
import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from data_utils import (
    RESULTS_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    PetDataset,
    load_classes
)


def main():
    model_path = MODEL_DIR / "pet_breed_resnet18.pth"
    test_csv = RESULTS_DIR / "test.csv"

    if not model_path.is_file():
        print("Model not found. Run: python src/train.py")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    classes, class_to_idx = load_classes()

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

    test_set = PetDataset(
        test_csv,
        class_to_idx,
        training=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    true_labels = []
    predictions = []

    # Predict test images without training
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predicted = outputs.argmax(1).cpu().tolist()

            predictions.extend(predicted)
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)

    # Precision, recall and F1-score for every breed
    report = classification_report(
        true_labels,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    # Create confusion matrix
    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(len(classes)))
    )

    plt.figure(figsize=(18, 16))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Oxford-IIIT Pet Test Confusion Matrix")
    plt.xlabel("Predicted breed")
    plt.ylabel("True breed")

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=90,
        fontsize=6
    )

    plt.yticks(
        range(len(classes)),
        classes,
        fontsize=6
    )

    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=180
    )

    plt.close()

    # Count wrong predictions by true/predicted breed pair
    pair_counts = {}

    for true_label, pred_label in zip(
        true_labels, predictions
    ):
        if true_label != pred_label:
            pair = (
                classes[true_label],
                classes[pred_label]
            )

            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    top_pairs = sorted(
        pair_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )[:3]

    summary = {
        "test_accuracy": float(accuracy),
        "test_images": len(true_labels),
        "number_of_breeds": len(classes),
        "top_confusions": [
            {
                "true_breed": pair[0],
                "predicted_breed": pair[1],
                "count": count
            }
            for pair, count in top_pairs
        ]
    }

    with open(
        RESULTS_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nFINAL TEST RESULTS")
    print("------------------")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test images: {len(true_labels)}")

    print("\nPrecision, recall and F1-score:")
    print(
        report_df[
            ["precision", "recall", "f1-score", "support"]
        ].to_string()
    )

    print("\nTop 3 confusion directions:")

    if not top_pairs:
        print("No incorrect predictions.")
    else:
        for (true_breed, predicted_breed), count in top_pairs:
            print(
                f"{true_breed} predicted as "
                f"{predicted_breed}: {count} times"
            )

    print("\nResults saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
