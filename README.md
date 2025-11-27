# immo-eliza-ml

A streamlined machine-learning workflow for predicting Belgian real estate prices. It includes data cleaning, feature engineering, multiple model baselines, and a final tuned log-target XGBoost model.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Data Cleaning Pipeline](#-data-cleaning-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Model Development](#-model-development)
- [Project Structure](#%EF%B8%8F-project-structure)
- [Requirements](#-requirements)
- [How to Run](#how-to-run-it)
- [Future Improvements](#future-improvements)
- [Limitations](#%EF%B8%8F-limitations)
- [Contributors](#-contributors)

---

## 🔎 Project Overview

This project predicts real-estate prices in Belgium using structured tabular data.  
The focus is on:

- Clean, leak-free preprocessing  
- Postal-code–based spatial feature engineering  
- Evaluation of linear and ensemble models  
- A final robust **log-target XGBoost** model achieving the best stability and accuracy.

---

## 🧹 Data Cleaning Pipeline

The unified cleaning step (`enhanced_clean`) handles:

- Boolean normalization  
- Numeric parsing  
- Sanity checks on build year, living area, rooms  
- Removal of invalid price entries  
- Dropping noisy fields  
- **Removal of `locality_name`** to reduce cardinality and simplify modeling  

Output: `cleaned_v2.csv`

---
## 🧬 Feature Engineering

Located in `src/feature_engineering.py`.

Key engineered features include:

- `postal_prefix` (numeric extraction)
- Region and density groups derived from postal codes
- Build-year signals (age, decade, age flags)
- Province → Region mapping
- Boolean flag features for garden, terrace, swimming pool

Output: `feature_engineered.csv`

---
## 🔧 Preprocessing Pipeline

The preprocessing pipeline prepares the dataset for all models and ensures consistent, leak-free transformations.

**Main steps:**

- **Train/Test Split (80/20):**  
  The dataset is split once to keep evaluation consistent.

- **Outlier Removal (training only):**  
  IQR filtering is applied only to the training set for `price`, `living_area`, and `number_rooms` to avoid leaking information into the test set.

- **Column Detection:**  
  Numeric and categorical columns are automatically identified.  
  `postal_code` is always treated as categorical.

- **Numeric Pipeline:**  
  - Median imputation  
  - Standard scaling  

- **Categorical Pipeline:**  
  - Most frequent imputation  
  - One-Hot Encoding with unknown-category handling  

- **ColumnTransformer:**  
  The numeric and categorical pipelines are combined into a unified preprocessing block used by all models.

This ensures that every model receives clean, encoded, and scaled input data with no leakage between training and testing.

## 🤖 Model Development

Models evaluated:

- **Ridge / Lasso / ElasticNet** — weak generalization  
- **Random Forest** — moderate but unstable  
- **XGBoost (raw target)** — improved but high variance  
- **XGBoost (log-target)** — **best model**, stable and consistent (test R² ~0.65)

### 📊 Model Performance Comparison Tuned Linear regression VS Tuned XGBoost

| Model                     | MAE (Train) | RMSE (Train) | R² (Train) | MAE (Test)  | RMSE (Test)  | R² (Test) |
|---------------------------|-------------|--------------|------------|-------------|--------------|-----------|
| Ridge                     | 17,134.86   | 23,333.16    | 0.9653     | 86,478.20   | 168,409.36   | 0.5548    |
| Lasso                     | 45,320.05   | 56,236.46    | 0.7987     | 86,777.13   | 169,305.18   | 0.5500    |
| ElasticNet                | 31,640.48   | 42,982.00    | 0.8824     | 86,785.47   | 168,936.84   | 0.5520    |
| XGBoost (Tuned, No Val)   | 71,626.79   | 124,144.21   | 0.7837     | 81,431.27   | 157,652.82   | 0.6100    |




## 🗂️ Project Structure

```bash
immo-eliza-ml/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── cleaning.py
│   ├── preprocessing.py
│   ├── train_xgboost_log.py
│   ├── Lin_reg.py
├── data/
│   ├── raw/
│   ├── processed/
│
└── models/ 



```

## 📦 Requirements

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
## How to run it

### Install dependencies

```bash

pip install -r requirements.txt
```

## Future Improvements

- Hyperparameter tuning for the log-XGBoost model
- Additional feature engineering
- Cross-validation for more robust evaluation
- Testing alternative models (e.g., LightGBM, CatBoost)
## ⚠️ Limitations 

The model relies heavily on the quality and comprehensiveness of the input data. It does not account for market trends or economic conditions. The model's predictions are specific to Belgium and may not generalize well to other regions. 

## 👥 Contributors 
This project is part of AI & Data Science Bootcamp training at **</becode** and it was done by: 
- Welederufeal Tadege [LinkedIn](https://www.linkedin.com/in/) | [Github](https://github.com/welde2001-bot) 
under the supervision of AI & data science coach ***Vanessa Rivera Quinones***

