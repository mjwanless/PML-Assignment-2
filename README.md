# PML Assignment 2: Credit Default Prediction

This project builds, evaluates, and deploys multiple machine learning models to predict the likelihood of a customer defaulting on their credit payment. The workflow includes extensive feature engineering, model training, performance evaluation, hyperparameter tuning, and ensemble stacking.

## 🔍 Problem Overview

The dataset includes historical credit data, demographic info, and bill/payment behavior. The goal is to predict whether a user will default (`CreditDefault = 1`) based on these features.

---

## 📂 Repository Structure

```
PML-Assignment-2/
├── BinaryFolder/                      # Saved model binaries and scaler
│   ├── GradientBoosting_model.pkl
│   ├── LogisticRegression_model.pkl
│   ├── RandomForest_model.pkl
│   ├── StackedModel.pkl
│   ├── nn_model.h5
│   └── scaler.pkl
├── Visualizations/                    # Plots and model diagnostics
├── CreditDefault.csv                 # Raw dataset
├── CreditDefault_Mystery.csv         # Unlabeled dataset for inference
├── CreditDefault_Predictions.csv     # Final predictions output
├── test.csv                          # Validation split
├── train.csv                         # Training split
├── TrainingScript.py                 # Full training + tuning + evaluation pipeline
├── Production.py                     # Inference pipeline for test data
└── README.md                         # Project documentation
```

---

## ⚙️ Models Used

- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Deep Neural Network** (3 hidden layers, tuned via randomized search)
- **Stacked Model (Ensemble)** using Logistic Regression on top of base model outputs

---

## 📊 Features & Engineering

- Raw features: demographic data, bill amounts, payment statuses
- Engineered features:
  - `TOTAL_BILL`, `TOTAL_PAY`, `PAYMENT_RATIO`
  - `AVG_BILL_AMT`, `AVG_PAY_AMT`
  - `LATE_PAYMENT_COUNT`
- Scaling: `StandardScaler` applied to all numerical features

---

## 🧪 Model Evaluation

- Metric: **F1 Score**
- Evaluation: Stratified Train/Validation split with K-Fold CV for base models
- Plots generated:
  - ROC curves
  - Confusion matrices
  - Feature importance
  - Neural net training history
  - Stacked model weight visualizations
  - F1 comparison bar chart

---

## 🚀 How to Run

### 1. Clone & Setup

```bash
git clone https://github.com/mjwanless/PML-Assignment-2.git
cd PML-Assignment-2
pip install -r requirements.txt  # if available, otherwise install manually
```

### 2. Train Models

```bash
python TrainingScript.py
```

- Models will be saved to `BinaryFolder/`
- Visuals will be saved in `Visualizations/`

### 3. Generate Predictions

```bash
python Production.py
```

- Predictions for `CreditDefault_Mystery.csv` are saved to `CreditDefault_Predictions.csv`

---

## 📦 Dependencies (partial)

- `scikit-learn`
- `tensorflow` / `keras`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `joblib`

---

## 📜 License

This project is released under the MIT License.
