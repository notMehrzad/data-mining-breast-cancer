[![English](https://img.shields.io/badge/English-🇺🇸-blue)](README.md)
[![فارسی](https://img.shields.io/badge/فارسی-🇮🇷-green)](README.fa.md)

---

# Breast Cancer Classification

Classifying breast tumors as malignant or benign using machine learning

---

## Table of Contents

- [About The Project](#about-the-project)
  - [Dataset](#dataset)
  - [Models Used](#models-used)
- [Results](#results)
- [Model Performance Analysis](#model-performance-analysis)
  - [Data Split](#data-split)
  - [Best Performing Model](#best-performing-model-logistic-regression)
- [Visualizations](#visualizations)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## About The Project

This project uses the Breast Cancer Wisconsin dataset to build and compare
four different machine learning models. The goal is to predict whether a
tumor is malignant or benign based on 30 features from cell nuclei images.

### Dataset

- **Source**: scikit-learn's `load_breast_cancer()`
- **Samples**: 569
- **Features**: 30 (mean radius, texture, concavity, etc.)
- **Target classes**:
  - 0 = Malignant (cancerous)
  - 1 = Benign (non-cancerous)

### Models Used

- _Logistic Regression_
- _K-Nearest Neighbors (KNN)_
- _Support Vector Classifier (SVC)_
- _Decision Tree_

### Results

| Model               | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.9825   | 0.9861    | 0.9861 | 0.9861   | 0.9954  |
| KNN                 | 0.9825   | 0.9730    | 1.0000 | 0.9863   | 0.9823  |
| SVC                 | 0.9737   | 0.9859    | 0.9722 | 0.9790   | 0.9954  |
| Decision Tree       | 0.9211   | 0.9437    | 0.9306 | 0.9371   | 0.9177  |

> **Note**: _KNN_ achieved perfect recall (1.0000), meaning it caught every single malignant case in the test set - zero false negatives.

---

## Model Performance Analysis

### Data Split

- **Training set**: 455 samples (80% of data)
- **Testing set**: 114 samples (20% of data)
- **Split method**: Stratified random sampling (preserves class balance)
- **Random state**: 42 (for reproducible results)

### Best Performing Model: _Logistic Regression_

**Achieved:** 98.25% accuracy with 0.9954 AUC-ROC

**Why it performed best:**

- The breast cancer dataset is **linearly separable** to a large degree
- _Logistic Regression_ is well-suited for binary classification with numerical features
- Feature scaling helped the model converge properly

---

### Visualizations

![KNN Parameter Tuning plot](Figures/Figure_1.png)

- _KNN_ Parameter Tuning: Shows accuracy vs. k value to find the optimal number of neighbors

---

![ROC Curves plot](Figures/Figure_2.png)

- ROC Curves: Compares all models' ability to distinguish between classes

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/notMehrzad/data-mining-breast-cancer.git
cd data-mining-breast-cancer
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

The script will:

1. Load and explore the dataset

2. Find the best k value for _KNN_

3. Train all four models

4. Show performance metrics

5. Display ROC curves

---

## License

Licensed under the MIT License - see the [LICENSE](LICENSE) for details.
