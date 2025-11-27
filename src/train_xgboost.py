import pandas as pd
import numpy as np
import os

from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib


def group_split(df, group_col="locality_name"):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    groups = df[group_col]
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    return df.iloc[train_idx], df.iloc[test_idx]


def remove_train_outliers(X, y):
    df = X.copy()
    df["target"] = y

    # living area IQR
    if "living_area" in df.columns:
        Q1 = df["living_area"].quantile(0.25)
        Q3 = df["living_area"].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df["living_area"] >= Q1 - 1.5 * IQR) &
                (df["living_area"] <= Q3 + 1.5 * IQR)]

    if "number_rooms" in df.columns:
        df = df[df["number_rooms"].fillna(0) <= 12]

    y_clean = df["target"]
    X_clean = df.drop(columns=["target"])

    print("Train after outlier removal:", X_clean.shape)
    return X_clean, y_clean


def train_xgboost(df, target="price", save_path="models/xgboost_geo_tuned.pkl"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Split
    train_df, test_df = group_split(df, "locality_name")

    y_train_raw = train_df[target]
    y_test_raw = test_df[target]

    X_train_raw = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])

    # Outliers
    X_train, y_train_raw = remove_train_outliers(X_train_raw, y_train_raw)

    # log-transform
    y_train = np.log1p(y_train_raw)
    y_test = np.log1p(y_test_raw)

    # Categorical / numeric
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    param_dist = {
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__n_estimators": [400, 600, 900],
        "model__subsample": [0.6, 0.8],
        "model__colsample_bytree": [0.6, 0.8],
        "model__min_child_weight": [1, 5, 10],
        "model__gamma": [0, 1],
        "model__reg_lambda": [1, 3, 5]
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    preds_train = np.expm1(best_model.predict(X_train))
    preds_test = np.expm1(best_model.predict(X_test))

    mae_train = mean_absolute_error(y_train_raw, preds_train)
    rmse_train = np.sqrt(mean_squared_error(y_train_raw, preds_train))
    r2_train = r2_score(y_train_raw, preds_train)

    mae_test = mean_absolute_error(y_test_raw, preds_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_raw, preds_test))
    r2_test = r2_score(y_test_raw, preds_test)

    print("\n===== FINAL XGBOOST RESULTS =====")
    print("\n--- Train ---")
    print(f"MAE: {mae_train:,.2f}")
    print(f"RMSE: {rmse_train:,.2f}")
    print(f"R²: {r2_train:.4f}")

    print("\n--- Test ---")
    print(f"MAE: {mae_test:,.2f}")
    print(f"RMSE: {rmse_test:,.2f}")
    print(f"R²: {r2_test:.4f}")

    # Save model
    joblib.dump(best_model, save_path)
    print(f"Model saved to {save_path}")

    return best_model
