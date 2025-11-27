# train_xgboost_log.py

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
import os


def train_xgboost_log_target(
    df: pd.DataFrame,
    target: str = "price",
    save_path: str = "models/xgboost_log_model.pkl"
):
    """
    Train XGBoost using a log-transformed target.

    Workflow:
    - Remove missing target
    - Train/test split
    - log1p target transform
    - OneHotEncoder for categoricals
    - Passthrough for numericals
    - XGBoost model
    - Back-transform predictions using expm1
    - Save final pipeline (preprocessing + model)
    """

    # ----------------------------------------------------
    # 1. Remove missing target values
    # ----------------------------------------------------
    df = df[df[target].notna()].copy()

    # ----------------------------------------------------
    # 2. Define features and log-transformed target
    # ----------------------------------------------------
    X = df.drop(columns=[target])
    y = np.log1p(df[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ----------------------------------------------------
    # 3. Column selection
    # ----------------------------------------------------
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    # ----------------------------------------------------
    # 4. XGBoost Model Definition
    # ----------------------------------------------------
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist"
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # ----------------------------------------------------
    # 5. Train the Model
    # ----------------------------------------------------
    pipe.fit(X_train, y_train)

    # ----------------------------------------------------
    # 6. Predict and Back-Transform
    # ----------------------------------------------------
    train_preds = np.expm1(pipe.predict(X_train))
    test_preds = np.expm1(pipe.predict(X_test))

    y_train_real = np.expm1(y_train)
    y_test_real = np.expm1(y_test)

    # ----------------------------------------------------
    # 7. Evaluation Metrics
    # ----------------------------------------------------
    results = {
        "Train MAE": mean_absolute_error(y_train_real, train_preds),
        "Train RMSE": np.sqrt(mean_squared_error(y_train_real, train_preds)),
        "Train R2": r2_score(y_train_real, train_preds),

        "Test MAE": mean_absolute_error(y_test_real, test_preds),
        "Test RMSE": np.sqrt(mean_squared_error(y_test_real, test_preds)),
        "Test R2": r2_score(y_test_real, test_preds)
    }

    # ----------------------------------------------------
    # 8. Save Final Model Pipeline
    # ----------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipe, save_path)

    print(f"Model saved to: {save_path}")
    print("\n===== XGBoost (Log-Transformed Target) Results =====")
    for k, v in results.items():
        print(f"{k}: {v:,.2f}")

    return pipe, results


if __name__ == "__main__":
    # Example usage if run standalone:
    df_path = "data/processed/cleaned_v2.csv"
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        train_xgboost_log_target(df)
    else:
        print(f"Dataset not found: {df_path}")
