# immo-eliza-ml

A streamlined machine-learning workflow for predicting Belgian real estate prices. It includes data cleaning, feature engineering, multiple model baselines, and a final tuned log-target XGBoost model.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Cleaning Pipeline](#-cleaning-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Model Development](#-model-development)
- [Project Structure](#%EF%B8%8F-project-structure)
- [Requirements](#-requirements)
- [How to Run](#how-to-run)
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

## 🧹 Cleaning Pipeline

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

## 🤖 Model Development

Models evaluated:

- **Ridge / Lasso / ElasticNet** — weak generalization  
- **Random Forest** — moderate but unstable  
- **XGBoost (raw target)** — improved but high variance  
- **XGBoost (log-target)** — **best model**, stable and consistent (test R² ~0.65)



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
│
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

