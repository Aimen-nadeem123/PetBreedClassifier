
from pathlib import Path
import random
import re
import json

import pandas as pd


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "oxford-iiit-pet"
IMAGE_DIR = DATA_DIR / "images"
ANNOTATION_DIR = DATA_DIR / "annotations"
RESULTS_DIR = PROJECT_DIR / "results"

# Reproducible random split
random.seed(42)


def get_breed_name(image_name):
    """
    Example:
    Abyssinian_12 -> Abyssinian
    american_bulldog_5 -> american_bulldog
    """
    return re.sub(r"_\d+$", "", image_name)


def main():
    # Check that the dataset was extracted
    if not IMAGE_DIR.is_dir():
        print("ERROR: images folder was not found.")
        print(f"Expected location: {IMAGE_DIR}")
        print("Please extract images.tar.gz first.")
        return

    if not ANNOTATION_DIR.is_dir():
        print("ERROR: annotations folder was not found.")
        print(f"Expected location: {ANNOTATION_DIR}")
        print("Please extract annotations.tar.gz first.")
        return

    trainval_file = ANNOTATION_DIR / "trainval.txt"
    test_file = ANNOTATION_DIR / "test.txt"

    if not trainval_file.is_file() or not test_file.is_file():
        print("ERROR: trainval.txt or test.txt is missing.")
        print("Check that annotations.tar.gz was extracted correctly.")
        return

    records = []

    # Read the official train/validation list
    for split_file, original_split in [
        (trainval_file, "trainval"),
        (test_file, "test"),
    ]:
        with open(split_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                if len(parts) < 4:
                    continue

                image_name = parts[0]
                class_id = int(parts[1])

                filename = image_name + ".jpg"
                image_path = IMAGE_DIR / filename

                if not image_path.is_file():
                    continue

                breed = get_breed_name(image_name)

                records.append({
                    "filename": filename,
                    "label": breed,
                    "class_id": class_id,
                    "original_split": original_split,
                })

    if not records:
        print("ERROR: No images were found.")
        print("Check the location of your extracted images.")
        return

    df = pd.DataFrame(records)

    # Check that each official class ID matches one breed
    class_counts = df.groupby("class_id")["label"].nunique()

    if (class_counts > 1).any():
        print("ERROR: Some class IDs have multiple breed names.")
        return

    # Get the 37 breed names in alphabetical order
    classes = sorted(df["label"].unique().tolist())

    if len(classes) != 37:
        print(f"WARNING: Expected 37 breeds but found {len(classes)}.")

    # Keep the official test set separate
    trainval_df = df[df["original_split"] == "trainval"].copy()
    test_df = df[df["original_split"] == "test"].copy()

    # Split the official trainval data into train and validation
    train_rows = []
    val_rows = []

    for breed in classes:
        breed_rows = trainval_df[
            trainval_df["label"] == breed
        ].sample(frac=1, random_state=42)

        validation_count = max(1, round(len(breed_rows) * 0.2))

        val_rows.append(breed_rows.iloc[:validation_count])
        train_rows.append(breed_rows.iloc[validation_count:])

    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows)

    # Save only the columns needed by the other scripts
    output_columns = ["filename", "label"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df[output_columns].to_csv(
        RESULTS_DIR / "train.csv", index=False
    )
    val_df[output_columns].to_csv(
        RESULTS_DIR / "val.csv", index=False
    )
    test_df[output_columns].to_csv(
        RESULTS_DIR / "test.csv", index=False
    )

    with open(RESULTS_DIR / "classes.json", "w", encoding="utf-8") as file:
        json.dump(classes, file, indent=4)

    print("\nDataset preparation completed!")
    print(f"Total images found: {len(df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Testing images: {len(test_df)}")
    print(f"Number of breeds: {len(classes)}")
    print(f"\nOutput files saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
