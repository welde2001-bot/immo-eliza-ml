


# immo-eliza-ml


# Description 

practical workflow for preparing data, training linear and advanced regression models, evaluating performance, and optionally applying cross-validation and hyperparameter tuning

# 📑 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Cleaning Pipeline](#cleaning-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Key Insights](#key-insights)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

# 🔎 Project Overview

- This project aims to predict real estate prices in Belgium using various machine learning models. The primary objective is to provide accurate price estimates for properties based on their features like location, area, number of bedrooms, etc.

# 📊 Dataset

The dataset used in this project contains features about real estate properties in Belgium, including details such as property type, location, living area, number of bedrooms, and more. It comprises around 15,000 properties.

## Pipeline Overview

###  Enhanced Cleaning
Located in `src/enhanced_cleaning.py`.
- Handles missing values
- Fixes numeric fields
- Normalizes booleans
- Removes unrealistic values
- Reduces locality cardinality
- Generates region

## Feature Engineering
Located in `src/feature_engineering.py`.
- Build year features
- House age categories
- Boolean numeric flags
- Postal prefix
- Location-based enrichments

## XGBoost Training
Located in `src/train_xgboost.py`.
- Group-based locality split (prevents geospatial leakage)
- Train-only outlier removal
- Log-target modeling
- Hyperparameter tuning
- Saves final production model

## Main pipeline
Run everything:




# 🔢 

### Key steps:

1


# Exploratory Data Analysis





-

 # 🧬 Repo Structure

``` bash
immo-eliza-ml/
│
├── data/
│   ├── raw/
│   │   └── raw.csv
│   ├── processed/
│   │   ├── cleaned_v2.csv
│   │   └── feature_engineered.csv
│
├── models/
│   └── xgboost_geo_tuned.pkl
│
├── src/
│   ├── enhanced_cleaning.py
│   ├── feature_engineering.py
│   ├── train_xgboost.py
│   └── main.py
│
└── README.md
```

# 📦 Requirements

To run the full ImmoEliza ML pipeline, install the following Python packages:

```
pandas>=2.0.0
numpy>=1.23.0
scikit-learn>=1.4.0
xgboost>=2.0.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
python-dateutil>=2.8.2

```
# How to Run

## ⚙️ 1. Clone the repository



git clone https://github.com/welde2001-bot/immo-eliza-ml

## 🧩 2. Install dependencies

```  

pip install -r requirements.txt

```



## 📈 Performance

The best-performing model (XGBoost model) achieved an R² score of 0.83 on the test set, indicating that it can explain 83% of the variance in property prices.



python src/cleaning.py

## ⚠️ Limitations

The model relies heavily on the quality and comprehensiveness of the input data.
It does not account for market trends or economic conditions.
The model's predictions are specific to Belgium and may not generalize well to other regions.

# 👥 Contributors
This project is part of AI & Data Science Bootcamp training at **`</becode>`** and it was done by:

- Welederufeal Tadege [LinkedIn](https://www.linkedin.com/in/) | [Github](https://github.com/welde2001-bot)


under the supervision of AI & data science coach ***Vanessa Rivera Quinones***


