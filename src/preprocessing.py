# preprocessing.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def split_data(
    df: pd.DataFrame,
    target: str = "price",
    test_size: float = 0.20,
    random_state: int = 42,
):
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset.")

    df = df[df[target].notna()].copy()

    y = pd.to_numeric(df[target], errors="coerce")
    df = df[y.notna()].copy()
    y = y[y.notna()]

    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def remove_outliers_from_train(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train-only outlier removal (no test leakage).
    Uses IQR filtering on price and living_area (if present).
    """
    df_train = X_train.copy()
    df_train["price"] = y_train

    cols_to_filter = [c for c in ["price", "living_area"] if c in df_train.columns]

    def iqr_filter(df_: pd.DataFrame, col: str) -> pd.DataFrame:
        q1 = df_[col].quantile(0.25)
        q3 = df_[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return df_[(df_[col] >= lower) & (df_[col] <= upper)]

    for col in cols_to_filter:
        df_train = iqr_filter(df_train, col)

    if "number_rooms" in df_train.columns:
        df_train = df_train[df_train["number_rooms"].fillna(0) <= 12]

    y_train_clean = df_train["price"]
    X_train_clean = df_train.drop(columns=["price"])

    print("Training after outlier removal:", X_train_clean.shape)
    return X_train_clean, y_train_clean


def build_preprocessor(X_train: pd.DataFrame, drop_postal_code: bool = False) -> ColumnTransformer:
    """
    Numeric: median impute + StandardScaler
    Categorical: most_frequent impute + OneHotEncoder(ignore unknown)

    Set drop_postal_code=True for linear models if postal_code creates too many sparse columns.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    if "postal_code" in X_train.columns and "postal_code" not in categorical_cols:
        categorical_cols.append("postal_code")
    if "postal_code" in numeric_cols:
        numeric_cols.remove("postal_code")

    if drop_postal_code and "postal_code" in categorical_cols:
        categorical_cols.remove("postal_code")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    print("\nNumeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)
    print("drop_postal_code:", drop_postal_code)

    return preprocessor


def run_preprocessing_pipeline(
    path: str = "data/processed/feature_engineered.csv",
    target: str = "price",
    drop_postal_code: bool = False,
    remove_outliers: bool = False,
    test_size: float = 0.20,
    random_state: int = 42,
):
    df = pd.read_csv(path, dtype={"postal_code": "string"})

    X_train, X_test, y_train, y_test = split_data(
        df, target=target, test_size=test_size, random_state=random_state
    )

    if remove_outliers:
        X_train, y_train = remove_outliers_from_train(X_train, y_train)

    preprocessor = build_preprocessor(X_train, drop_postal_code=drop_postal_code)

    return X_train, X_test, y_train, y_test, preprocessor
