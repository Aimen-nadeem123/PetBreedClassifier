# Pet Breed Classifier

## About the Project

Pet Breed Classifier is a project that identifies the breed of a cat or dog from an image. It uses a pretrained ResNet-18 model to classify images into 37 different pet breeds.

The main purpose of this project is to learn how image classification works, train a model using an existing dataset, evaluate its performance, and check how well it works on images outside the main dataset.

## Features

- Classifies images into 37 cat and dog breeds.
- Uses a pretrained ResNet-18 model.
- Trains and validates the model using separate datasets.
- Evaluates the model on test images.
- Calculates accuracy, precision, recall, and a confusion matrix.
- Shows the breed pairs that the model confuses most often.
- Tests the model on 30 additional external images.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## Dataset

This project uses the Oxford-IIIT Pet dataset, which contains images of cats and dogs belonging to 37 breeds.

The dataset images are organized into training, validation, and testing sets. The training set is used to train the model, the validation set helps monitor its performance during training, and the test set is used for final evaluation.

### Dataset Split

| Dataset | Number of images |
|---|---:|
| Training | 2,944 |
| Validation | 736 |
| Testing | 3,669 |
| **Total** | **7,349** |

## Model Used

I used ResNet-18, a convolutional neural network that has already been trained on ImageNet images.

Instead of training the complete model from scratch, I started with its pretrained weights and adapted the model for the 37 pet breeds. This approach is called "transfer learning".

The model was trained for 8 epochs. The best validation accuracy was 90.35%.

## Project Structure

PetBreedClassifier/
│
├── data/
│   └── oxford-iiit-pet/
│
├── models/
│   └── pet_breed_resnet18.pth
│
├── results/
│   └── external_predictions.csv
│
├── src/
│   ├── prepare_data.py
│   ├── data_utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── external_eval.py
│
├── .gitignore
├── requirements.txt
└── README.md


The `data` folder contains the dataset, `src` contains the Python scripts, `models` stores the trained model, and `results` contains evaluation output.

## Installation

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/PetBreedClassifier.git
cd PetBreedClassifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the Oxford-IIIT Pet dataset and place its images and annotations in the folder expected by the data preparation script.

Then run:

```bash
python src/prepare_data.py
```

### 5. Train the model

```bash
python src/train.py
```

The trained model is saved in the `models` folder.

### 6. Evaluate the model

```bash
python src/evaluate.py
```

This script evaluates the model on the test dataset and generates evaluation results.

### 7. Run external evaluation

Place the 30 external images in the folder expected by `src/external_eval.py`, then run:

```bash
python src/external_eval.py
```

The predictions are saved to `results/external_predictions.csv`.

## Results

### Main Test Results

| Metric | Result |
|---|---:|
| Number of breeds | 37 |
| Training epochs | 8 |
| Best validation accuracy | 90.35% |
| Test accuracy | 86.86% |
| Test images | 3,669 |

The model achieved 90.35% accuracy on the validation set and 86.86% accuracy on the test set. These results show that it learned to recognize many of the pet breeds, although some breeds were difficult to distinguish.

### Most Confused Breed Pairs

The following were the three most frequent incorrect predictions in the test results.

| Actual breed | Predicted breed | Incorrect predictions |
|---|---|---:|
| Staffordshire Bull Terrier | American Pit Bull Terrier | 27 |
| Birman | Ragdoll | 26 |
| Basset Hound | Beagle | 17 |

These results suggest that the model sometimes confuses breeds with similar physical features.

### External Evaluation

I also evaluated the model on 30 additional images.

| Metric | Result |
|---|---:|
| External images | 30 |
| Correct predictions | 7 |
| Incorrect predictions | 23 |
| External accuracy | 23.33% |

The external accuracy was lower than the main test accuracy. This shows that the model did not perform as well on these additional images.

## Limitations

- Some breeds have similar appearances, which can lead to incorrect predictions.
- Performance may change with different lighting, backgrounds, image quality, and camera angles.
- The external evaluation contains only 30 images.
- The model's test accuracy does not guarantee the same accuracy on images from every source.

## Future Improvements

In the future, I would like to:

- Collect more correctly labelled images from different sources.
- Improve performance on external images.
- Use data augmentation to help the model handle different image conditions.
- Investigate the breeds that are most frequently confused.
- Fine-tune the model further and compare its performance with other pretrained models.
- Evaluate the model on a larger and more diverse external dataset.

## Conclusion

This project helped me understand image classification, transfer learning, model training, and performance evaluation using Python and PyTorch. The model achieved 86.86% test accuracy, but the external evaluation showed that there is still room for improvement. Improving generalization and testing on more diverse images are important next steps.

## AIMEN NADEEM