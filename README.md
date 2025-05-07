# Smoking Detection - Classification Based on Biological Data

Data is collected from Dataset: https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking.

## Overview

This project aims to identify smokers based on biological data that have been processed to detect significant patterns. The dataset consists of 27 variables, including both categorical (e.g., gender, presence of tartar) and numerical (e.g., glucose levels) features. The goal is to build a classification model that can accurately predict whether an individual is a smoker.

The project employs various approaches for data preprocessing and modeling, including:
- encoding categorical variables using **one-hot encoding**.
- using **undersampling** and **oversampling** techniques to balance the dataset for imbalanced class distributions.
- training different classification models, such as neural networks and traditional machine learning algorithms.

## Requirements

To run the scripts and replicate the analysis, ensure the following dependencies are installed:

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow` (for deep learning models)
- `matplotlib`
- `seaborn`

These can be installed via `pip`:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
````

## Data Preprocessing

The dataset consists of both categorical and numerical features. Categorical features are encoded using one-hot encoding, while numerical features are scaled appropriately using standard scaling techniques. Additionally, due to imbalanced class distribution (smokers vs. non-smokers), techniques like undersampling and oversampling are applied to ensure the model does not bias towards the majority class.

### Categorical Variables

Categorical variables are encoded using **one-hot encoding** to convert them into a numerical format that can be processed by machine learning models.

### Handling Imbalanced Data

Two common techniques, **undersampling** and **oversampling**, are used:

* **Undersampling**: Reduces the number of samples from the majority class (non-smokers) to balance the dataset.
* **Oversampling**: Increases the number of samples from the minority class (smokers) using methods like SMOTE (Synthetic Minority Over-sampling Technique).

## Model Training

Various classification models are evaluated during the training process, including:

* **Neural Networks (DNN)** - a deep learning model to predict whether an individual is a smoker.
* **Logistic Regression** - a baseline linear model.
* **Random Forest** - an ensemble model using decision trees.
* **Support Vector Machine (SVM)** - a linear and non-linear classifier.
* **K-Nearest Neighbors (KNN)** - simple classifier.

Each model is evaluated based on performance metrics such as accuracy, precision, recall, and F1 score. Hyperparameter tuning and cross-validation are applied to optimize the models.

## Model Evaluation

The evaluation phase involves assessing model performance using various metrics:

* **accuracy** - the percentage of correctly predicted instances.
* **precision** - the percentage of correct positive predictions among all positive predictions.
* **recall** - the percentage of correct positive predictions among all actual positives.
* **F1 score** - the harmonic mean of precision and recall.

A comparison of these metrics across different models is presented to determine the best-performing classifier for the smoking detection task.

## Results

After training and evaluating several models, the best model is selected based on the highest performance metrics, particularly accuracy, precision, and recall. The model with the best balance of these metrics is then used to predict whether new individuals are smokers based on their biological data.
