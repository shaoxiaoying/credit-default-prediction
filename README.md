# credit-default-prediction
Credit Risk Modelling: Customer Default Prediction with Machine Learning
# Credit Risk Modelling: Customer Default Prediction

## Project Overview
This repository implements an end-to-end Machine Learning solution for predicting credit default risk on credit application data. The goal is to classify borrowers as low risk (`good`) or high risk (`bad`) to optimize underwriting decisions and manage non-performing loans (NPLs).

## Dataset
* **Source**: German Credit Data
* **Observations**: 1,000 loans
* **Features**: 20 variables (e.g., `status_account`, `credit_amount`, `month_duration`, `credit_history`, `years_employment`)
* **Target**: Binary classification (`0` = Good / Non-default, `1` = Bad / Default)

## Key Results
* **ROC-AUC Score**: ~0.807 (Logistic Regression with class rebalancing)
* **Default Recall**: 80% default detection rate on holdout test data

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/credit-risk-modelling.git](https://github.com/your-username/credit-risk-modelling.git)
   cd credit-risk-modelling
