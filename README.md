# Practical Application III – Comparing Classifiers

This project implements and compares several supervised classification models on a bank marketing dataset. The task is to predict whether a customer will subscribe to a term deposit (`y` = yes/no) using demographic, campaign, and macroeconomic features.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Goals](#project-goals)
- [Methods](#methods)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

The notebook walks through a typical end‑to‑end ML workflow:

- Load and inspect the **bank-additional-full.csv** dataset.
- Explore and visualize features (client, campaign, and economic attributes).
- Encode categorical variables, scale numeric variables, and split into train/test sets.
- Train baseline and improved classifiers.
- Compare models on accuracy, precision, recall, and training time.

## Dataset

- **Source**: UCI Bank Marketing dataset (bank-additional-full.csv).
- **Rows**: ~41k examples.
- **Target variable**: `y` – whether the client subscribed to a term deposit (`yes`/`no`).
- **Feature groups** (examples):
  - Client: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`.
  - Contact / campaign: `contact`, `month`, `day_of_week`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`.
  - Economic: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`.
- The dataset is imbalanced (far more `no` than `yes`).

## Project Goals

- Define a clear **business objective**: identify clients likely to subscribe to a term deposit so marketing efforts can be better targeted.
- Build multiple classifiers and compare them:
  - Logistic Regression
  - K‑Nearest Neighbors (KNN)
  - Decision Tree
  - Support Vector Machine (SVM)
- Evaluate and discuss trade‑offs between:
  - Accuracy
  - Precision / Recall for the positive class
  - Train time and complexity
- Explore techniques to handle class imbalance and improve model performance.

## Methods

### Tooling

- Python, Jupyter Notebook
- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `plotly.express`
- `scikit-learn` (preprocessing, models, metrics, `Pipeline`, `ColumnTransformer`, `GridSearchCV`)

### Preprocessing

- Train/test split with stratification on `y`.
- Categorical features: one‑hot encoded.
- Numeric features: standardized with `StandardScaler`.
- Combined using a `ColumnTransformer` inside a `Pipeline`.

### Models

Baseline models (default hyperparameters first):

- Logistic Regression
- K‑Nearest Neighbors
- Decision Tree
- Support Vector Classifier (SVC)

Model improvements:

- **Class‑weighted Logistic Regression** (`class_weight="balanced"`) to address imbalance and improve recall.
- **Decision Tree hyperparameter tuning** via `GridSearchCV` (e.g., `max_depth`, `min_samples_split`, `min_samples_leaf`, optimized for F1).

### Evaluation

For each model, the notebook typically records:

- Training time
- Train accuracy
- Test accuracy
- Precision, recall, and F1 for the positive class
- Confusion matrix and supporting plots where useful

## Results Summary

High‑level observations (you can adjust numbers if you logged them explicitly):

- Baseline Logistic Regression achieves strong accuracy but modest recall on the minority (positive) class.
- SVM performs competitively but is more computationally expensive to train.
- Decision Trees provide interpretability but may overfit without tuning.
- Class‑weighted Logistic Regression significantly improves recall for `y = yes` while reducing overall accuracy.
- Tuned Decision Trees improve F1 over the default tree but still trade off against the balanced Logistic Regression in recall.

## Repository Structure

A suggested structure:
├── README.md # Project documentation
├── prompt_III.ipynb # Main analysis notebook
├── data/
│ └── bank-additional-full.csv
├── images/ # (Optional) Saved plots from EDA / modeling
│ └── *.png
└── requirements.txt # Python dependencies (optional)


## Setup and Installation

1. Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate # macOS / Linux
venv\Scripts\activate # Windows


2. Install dependencies (either from `requirements.txt` if you create one, or minimally):

pip install pandas numpy matplotlib seaborn plotly scikit-learn


3. Ensure the dataset file `bank-additional-full.csv` is located under `data/` at the project root.

## How to Run

1. Start Jupyter:

jupyter notebook


2. Open `prompt_III.ipynb`.
3. Run all cells in order:
- Data loading and exploration
- Feature engineering and preprocessing
- Baseline model training and evaluation
- Model comparison and tuning experiments

## Future Improvements

- Add cross‑validation and more robust model selection.
- Try additional models (Random Forest, Gradient Boosting, XGBoost, etc.).
- Implement custom decision thresholds and ROC/PR curve analysis.
- Package the best model into a simple API or CLI for batch scoring.

## License

Specify a license here (e.g., MIT, Apache 2.0) or “All rights reserved” depending on how you intend to share this work.

