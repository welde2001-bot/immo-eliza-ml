# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# 1. TRAIN / TEST SPLIT (80 / 20)
# ------------------------------------------------------------
def split_data(df: pd.DataFrame, target: str = "price"):
    """
    Split dataset into train and test sets (80/20).

    Parameters
    ----------
    df : DataFrame
        Input dataset after feature engineering.
    target : str
        Target column name.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    y = df[target]
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# 2. OUTLIER REMOVAL FROM TRAINING SET ONLY
# ------------------------------------------------------------
def remove_outliers_from_train(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Remove outliers ONLY from the training dataset
    using IQR filtering on selected columns.

    Avoids any test leakage.

    Returns
    -------
    X_train_clean, y_train_clean
    """

    df_train = X_train.copy()
    df_train["price"] = y_train

    cols_to_filter = [
        col for col in ["price", "living_area"]
        if col in df_train.columns
    ]

    def iqr_filter(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[col] >= lower) & (df[col] <= upper)]

    for col in cols_to_filter:
        df_train = iqr_filter(df_train, col)

    # Specific rule: number_rooms <= 12
    if "number_rooms" in df_train.columns:
        df_train = df_train[df_train["number_rooms"].fillna(0) <= 12]

    y_train_clean = df_train["price"]
    X_train_clean = df_train.drop(columns=["price"])

    print("Training after outlier removal:", X_train_clean.shape)

    return X_train_clean, y_train_clean


# ------------------------------------------------------------
# 3. BUILD PREPROCESSOR (numeric + categorical)
# ------------------------------------------------------------
def build_preprocessor(X_train: pd.DataFrame):
    """
    Build a sklearn ColumnTransformer for numerical and categorical features.

    Ensures:
    - Numeric: impute median + StandardScaler
    - Categorical: impute most frequent + OneHotEncoder
    - postal_code is always treated as categorical

    Returns
    -------
    preprocessor : ColumnTransformer
    """

    numeric_cols = X_train.select_dtypes(
        include=["float64", "int64", "Int64"]
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # Ensure postal_code remains categorical
    if "postal_code" in numeric_cols:
        numeric_cols.remove("postal_code")
        categorical_cols.append("postal_code")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    print("\nNumeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    return preprocessor


# ------------------------------------------------------------
# 4. MAIN PREPROCESSING PIPELINE FUNCTION
# ------------------------------------------------------------
def run_preprocessing_pipeline(
    path: str = "data/processed/feature_engineered.csv",
    target: str = "price"
):
    """
    Load engineered dataset, enforce postal_code dtype,
    split 80/20, remove outliers from training only,
    and build preprocessing pipeline.

    Returns
    -------
    X_train_clean, X_test, y_train_clean, y_test, preprocessor
    """

    df = pd.read_csv(path, dtype={"postal_code": "string"})

    # Split
    X_train, X_test, y_train, y_test = split_data(df, target=target)

    # Outliers removed ONLY from train
    X_train_clean, y_train_clean = remove_outliers_from_train(X_train, y_train)

    # Preprocessor
    preprocessor = build_preprocessor(X_train_clean)

    return X_train_clean, X_test, y_train_clean, y_test, preprocessor
