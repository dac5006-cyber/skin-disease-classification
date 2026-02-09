# Skin Disease Classification (MATLAB)

This repository provides the MATLAB implementation used in the research
on skin disease classification.

## Methodology Overview
The proposed approach extracts deep features using a pre-trained
ResNet-50 network and performs binary classification using a linear
Support Vector Machine (SVM). Hyperparameter tuning is conducted using
five-fold cross-validation on the training set only.

## Dataset Structure
The code assumes the dataset is organized as follows:

- train/
- valid/
- test/

Each folder contains subfolders corresponding to class labels.

## Notes
- Trained model weights and datasets are not included.
- The code is shared for methodological transparency and academic use.

## License
MIT License.
