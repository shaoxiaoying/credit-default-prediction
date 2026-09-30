# Credit Risk Modelling: Customer Default Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline designed to assess credit risk and predict borrower default probability on financial application data.

---

## 📌 Business Overview
In financial lending, assessing borrower creditworthiness is critical to maintaining a healthy loan portfolio. False negatives (classifying a high-risk borrower as low-risk) lead to direct monetary loss via non-performing loans (NPLs). 

This project builds a binary classification pipeline to predict customer default risk (`bad` vs. `good` credit) using historical financial, demographic, and transactional attributes.

---

## 📊 Dataset Summary
The dataset (`german_credit_data.csv`) consists of **1,000 observations** across **20 feature attributes**:

* **Target Variable**: `target`
  * `0` (`good`): Low-risk borrower (700 observations / 70%)
  * `1` (`bad`): High-risk / default borrower (300 observations / 30%)
* **Numerical Features**: `credit_amount`, `month_duration`, `age`, `payment_to_income_ratio`, `residence_since`, `n_credits`, `n_guarantors`
* **Categorical Features**: `status_account`, `credit_history`, `purpose`, `status_savings`, `years_employment`, `status_and_sex`, `secondary_obligor`, `collateral`, `other_installment_plans`, `housing`, `job`, `telephone`, `is_foreign_worker`

---

## 🛠️ Methodology & Architecture

1. **Preprocessing & Encoding**:
   * Numeric features are standard-scaled (`StandardScaler`).
   * Categorical values are one-hot encoded (`OneHotEncoder`) with drop-first logic to avoid collinearity.
2. **Handling Class Imbalance**:
   * Cost-sensitive learning (`class_weight='balanced'`) is applied to penalize false negatives higher than false positives.
3. **Model Selection**:
   * Evaluated **Logistic Regression** (baseline/interpretable) and **Random Forest Classifier** (ensemble non-linear baseline).

---

## 📈 Key Results

| Model | ROC-AUC | Default Precision (1) | Default Recall (1) | F1-Score (1) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **0.8067** | **0.55** | **0.80** | **0.65** |
| **Random Forest (Balanced)** | **0.8021** | -- | -- | -- |

* **Key Finding**: Logistic Regression with class-weight balancing achieved an **80% recall rate** on customer defaults with an overall **ROC-AUC of 0.807**, making it a strong, highly interpretable model for credit underwriting.

---

## 🚀 Quick Start

### 1. Requirements
* Python 3.9+
* `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`

### 2. Installation & Setup
```bash
# Clone repository
git clone [https://github.com/your-username/credit-risk-modelling.git](https://github.com/your-username/credit-risk-modelling.git)
cd credit-risk-modelling

# Install dependencies
pip install -r requirements.txt

3. Pipeline Execution
Bash
# Train the model pipeline
python src/train.py

# Run model evaluation metrics
python src/evaluate.py
