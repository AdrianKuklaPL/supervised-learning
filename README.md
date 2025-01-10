# Supervised Learning Analysis

## Objective
This repository contains analyses of two datasets using supervised learning algorithms: K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Multi-Layer Perceptron (MLP) Classifier (Neural Network). The project focuses on creating validation and learning curves, evaluating algorithm performance, and comparing models.

## Technologies Used
Python

## Analyses Performed
- Model evaluation using validation curves and learning curves.
- Supervised learning algorithms explored:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVM)
  - Multi-Layer Perceptron (MLP) Classifier
- Hyperparameter tuning for each algorithm using GridSearchCV.
- Evaluation metrics:
  - Accuracy
  - F1-score
  - Precision
  - Recall
- Oversampling using SMOTE for handling class imbalance.
- Visualizations of model performance and learning trends.

## Libraries Used
The following libraries are required and can be installed via `pip`:
- `ucimlrepo`
- `numpy`
- `pandas`
- `random`
- `matplotlib`
- `sklearn`
- `imblearn`
- `time`

## Datasets
The following datasets from the UCI Machine Learning Repository were used in this analysis:
- [Drug Consumption (Quantified) Dataset](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)
- [Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)

## How to Replicate
1. Clone the repository:
   ```bash
   git clone https://github.com/AdrianKuklaPL/supervised-learning.git
